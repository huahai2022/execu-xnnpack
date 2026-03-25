/*
 * CPU-only SmolVLM multimodal runner for Android with explicit TTFT/prefill
 * /decode metrics.
 */

#include <cinttypes>
#include <fstream>
#include <memory>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <gflags/gflags.h>

#include <executorch/extension/llm/runner/io_manager/io_manager.h>
#include <executorch/extension/llm/runner/llm_runner_helper.h>
#include <executorch/extension/llm/runner/multimodal_decoder_runner.h>
#include <executorch/extension/llm/runner/multimodal_input.h>
#include <executorch/extension/llm/runner/multimodal_prefiller.h>
#include <executorch/extension/llm/runner/multimodal_runner.h>
#include <executorch/extension/llm/runner/constants.h>
#include <executorch/extension/llm/runner/stats.h>
#include <executorch/extension/llm/runner/text_token_generator.h>
#include <executorch/extension/llm/runner/util.h>
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/platform/log.h>

#if defined(ET_USE_THREADPOOL)
#include <executorch/extension/threadpool/cpuinfo_utils.h>
#include <executorch/extension/threadpool/threadpool.h>
#endif

DEFINE_string(model_path, "smolvlm_cpu.pte", "Path to .pte model.");
DEFINE_string(tokenizer_path, "tokenizer.json", "Path to tokenizer JSON.");
DEFINE_string(
    image_bin_path,
    "image_512_chw_f32.bin",
    "Path to raw float32 CHW image tensor.");
DEFINE_string(prompt, "Can you describe this image?", "Prompt text.");
DEFINE_double(temperature, 0.0f, "Sampling temperature.");
DEFINE_int32(
    max_new_tokens,
    2,
    "Maximum new tokens. Use >=2 to report a decode speed.");
DEFINE_int32(cpu_threads, -1, "CPU thread count.");
DEFINE_bool(warmup, false, "Run one warmup generation before the measured run.");

namespace {

using ::executorch::extension::llm::GenerationConfig;
using ::executorch::extension::llm::Image;
using ::executorch::extension::llm::IOManager;
using ::executorch::extension::llm::kEnableDynamicShape;
using ::executorch::extension::llm::kMaxContextLen;
using ::executorch::extension::llm::kTextModelMethod;
using ::executorch::extension::llm::kTokenEmbeddingMethod;
using ::executorch::extension::llm::kUseKVCache;
using ::executorch::extension::llm::kVisionEncoderMethod;
using ::executorch::extension::llm::make_image_input;
using ::executorch::extension::llm::make_text_input;
using ::executorch::extension::llm::MultimodalDecoderRunner;
using ::executorch::extension::llm::MultimodalInput;
using ::executorch::extension::llm::MultimodalPrefiller;
using ::executorch::extension::llm::MultimodalRunner;
using ::executorch::extension::llm::Stats;
using ::executorch::extension::llm::TextTokenGenerator;
using ::executorch::extension::llm::time_in_ms;
using ::executorch::extension::Module;

constexpr int kImageSize = 512;
constexpr int kChannels = 3;

MultimodalInput load_image_bin(const std::string& image_path) {
  const size_t numel = static_cast<size_t>(kChannels) * kImageSize * kImageSize;
  const std::streamsize expected_bytes =
      static_cast<std::streamsize>(numel * sizeof(float));

  std::ifstream file(image_path, std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    ET_LOG(Error, "Failed to open image bin: %s", image_path.c_str());
    throw std::runtime_error("failed to open image bin");
  }

  const std::streamsize size = file.tellg();
  if (size != expected_bytes) {
    ET_LOG(
        Error,
        "Unexpected image bin size: got %ld bytes, expected %ld",
        static_cast<long>(size),
        static_cast<long>(expected_bytes));
    throw std::runtime_error("unexpected image bin size");
  }

  file.seekg(0, std::ios::beg);
  std::vector<float> buffer(numel);
  file.read(reinterpret_cast<char*>(buffer.data()), expected_bytes);
  file.close();

  Image image(std::move(buffer), kImageSize, kImageSize, kChannels);
  return make_image_input(std::move(image));
}

class InstrumentedMultimodalRunner final : public MultimodalRunner {
 public:
  using MultimodalRunner::MultimodalRunner;

  ::executorch::runtime::Error benchmark(
      const std::vector<MultimodalInput>& inputs,
      const GenerationConfig& config) {
    if (inputs.empty()) {
      ET_LOG(Error, "MultimodalInput vector cannot be empty");
      return ::executorch::runtime::Error::InvalidArgument;
    }

    if (!is_loaded()) {
      ET_CHECK_OK_OR_RETURN_ERROR(load());
    }

    stats_->reset();
    pos_ = 0;

    const auto callback = [](const std::string& piece) {
      ::executorch::extension::llm::safe_printf(piece.c_str());
      fflush(stdout);
    };

    const long inference_start_ms = time_in_ms();
    uint64_t first_token = 0;
    for (size_t i = 0; i < inputs.size(); ++i) {
      const auto& input = inputs[i];
      ::executorch::runtime::EValue encoder_output;
      long stage_start_ms = time_in_ms();

      if (input.is_text() || input.is_tokens()) {
        std::vector<uint64_t> tokens;
        if (input.is_text()) {
          auto encode_result = tokenizer_->encode(
              input.get_text(),
              (i == 0 && pos_ == 0) ? config.num_bos : 0,
              (i == 0 && pos_ == 0) ? config.num_eos : 0);
          if (!encode_result.ok()) {
            ET_LOG(
                Error,
                "Tokenizers error code %d",
                static_cast<uint32_t>(encode_result.error()));
            return ::executorch::runtime::Error::InvalidArgument;
          }
          tokens = std::move(*encode_result);
        } else {
          tokens = input.get_tokens();
        }
        const long tokenize_end_ms = time_in_ms();

        auto text_tensor = ::executorch::extension::from_blob(
            tokens.data(),
            {1, static_cast<executorch::aten::SizesType>(tokens.size())},
            ::executorch::aten::ScalarType::Long);
        auto token_embedding_result =
            module_->execute(kTokenEmbeddingMethod, text_tensor);
        ET_CHECK_OK_OR_RETURN_ERROR(token_embedding_result.error());
        const long embedding_end_ms = time_in_ms();
        encoder_output = token_embedding_result.get()[0];

        std::vector<int64_t> cache_positions;
        const int64_t seq_len = encoder_output.toTensor().size(1);
        auto cache_position_result =
            ::executorch::extension::llm::populate_start_pos_or_cache_position(
                module_.get(),
                pos_,
                cache_positions,
                static_cast<int>(seq_len),
                kTextModelMethod);
        ET_CHECK_OK_OR_RETURN_ERROR(cache_position_result.error());
        auto prefill_result = module_->execute(
            kTextModelMethod, {encoder_output, cache_position_result.get()});
        ET_CHECK_OK_OR_RETURN_ERROR(prefill_result.error());
        const long decoder_end_ms = time_in_ms();
        auto outputs_res = prefill_result.get()[0].toTensor();
        pos_ += seq_len;
        first_token = static_cast<uint64_t>(
            text_decoder_runner_->logits_to_token(outputs_res));
        ET_LOG(
            Info,
            "Segment %zu text: tokens=%zu tokenize=%.3f s embed=%.3f s decoder_prefill=%.3f s",
            i,
            tokens.size(),
            (tokenize_end_ms - stage_start_ms) / 1000.0,
            (embedding_end_ms - tokenize_end_ms) / 1000.0,
            (decoder_end_ms - embedding_end_ms) / 1000.0);
        continue;
      }

      if (input.is_image()) {
        const Image& image = input.get_image();
        auto image_tensor_result = image.toTensor(/*with_batch=*/true);
        ET_CHECK_OK_OR_RETURN_ERROR(image_tensor_result.error());
        auto image_tensor = image_tensor_result.get();

        ET_LOG(
            Info,
            "Image tensor dim: %zu, dtype: %s",
            image_tensor->dim(),
            ::executorch::runtime::toString(image_tensor->scalar_type()));

        auto image_encoder_result =
            module_->execute(kVisionEncoderMethod, image_tensor);
        ET_CHECK_OK_OR_RETURN_ERROR(image_encoder_result.error());
        const long image_encoder_end_ms = time_in_ms();
        encoder_output = image_encoder_result.get()[0];

        std::vector<int64_t> cache_positions;
        const int64_t seq_len = encoder_output.toTensor().size(1);
        ET_LOG(
            Info,
            "Segment %zu image: vision_encoder=%.3f s image_seq_len=%" PRId64,
            i,
            (image_encoder_end_ms - stage_start_ms) / 1000.0,
            seq_len);
        auto cache_position_result =
            ::executorch::extension::llm::populate_start_pos_or_cache_position(
                module_.get(),
                pos_,
                cache_positions,
                static_cast<int>(seq_len),
                kTextModelMethod);
        ET_CHECK_OK_OR_RETURN_ERROR(cache_position_result.error());
        auto prefill_result = module_->execute(
            kTextModelMethod, {encoder_output, cache_position_result.get()});
        ET_CHECK_OK_OR_RETURN_ERROR(prefill_result.error());
        const long decoder_end_ms = time_in_ms();
        auto outputs_res = prefill_result.get()[0].toTensor();
        pos_ += seq_len;
        first_token = static_cast<uint64_t>(
            text_decoder_runner_->logits_to_token(outputs_res));
        ET_LOG(
            Info,
            "Segment %zu image: decoder_prefill=%.3f s",
            i,
            (decoder_end_ms - image_encoder_end_ms) / 1000.0);
        continue;
      }

      ET_LOG(Error, "Unsupported multimodal input");
      return ::executorch::runtime::Error::NotSupported;
    }

    const long first_token_ms = time_in_ms();

    const int64_t prompt_tokens = pos_;

    auto first_piece = tokenizer_->decode(first_token, first_token);
    if (!first_piece.ok()) {
      ET_LOG(
          Error,
          "Tokenizers error code %d",
          static_cast<uint32_t>(first_piece.error()));
      return ::executorch::runtime::Error::InvalidArgument;
    }
    callback(std::move(*first_piece));

    const long ttft_ms = first_token_ms - inference_start_ms;
    const double prefill_tokens_per_s =
        prompt_tokens > 0 && ttft_ms > 0
        ? static_cast<double>(prompt_tokens) * 1000.0 /
            static_cast<double>(ttft_ms)
        : 0.0;

    const int32_t max_context_len = metadata_.at(kMaxContextLen);
    const int32_t resolved_max_new_tokens =
        config.resolve_max_new_tokens(max_context_len, pos_);

    ET_LOG(
        Info,
        "Resolved max_new_tokens=%d prompt_tokens=%" PRId64
        " max_context_len=%d",
        resolved_max_new_tokens,
        prompt_tokens,
        max_context_len);

    int64_t extra_tokens = 0;
    long decode_end_ms = first_token_ms;
    if (resolved_max_new_tokens > 1) {
      text_token_generator_->set_ignore_eos(config.ignore_eos);
      auto generate_result = text_token_generator_->generate(
          /*tokens=*/{first_token},
          /*start_pos=*/pos_,
          /*max_new_tokens=*/resolved_max_new_tokens - 1,
          /*temperature=*/config.temperature,
          /*token_callback=*/callback);
      if (!generate_result.ok()) {
        return generate_result.error();
      }
      extra_tokens = generate_result.get();
      decode_end_ms = time_in_ms();
      pos_ += extra_tokens;
    }

    printf("\n");

    const long decode_ms = decode_end_ms - first_token_ms;
    const double decode_tokens_per_s =
        extra_tokens > 0 && decode_ms > 0
        ? static_cast<double>(extra_tokens) * 1000.0 /
            static_cast<double>(decode_ms)
        : 0.0;
    const double end_to_end_tokens_per_s =
        (1 + extra_tokens) > 0 && (decode_end_ms - inference_start_ms) > 0
        ? static_cast<double>(1 + extra_tokens) * 1000.0 /
            static_cast<double>(decode_end_ms - inference_start_ms)
        : 0.0;

    ET_LOG(Info, "===== Benchmark =====");
    ET_LOG(
        Info,
        "Prompt tokens (including multimodal prefill): %" PRId64,
        prompt_tokens);
    ET_LOG(Info, "TTFT: %.3f s", ttft_ms / 1000.0);
    ET_LOG(
        Info,
        "Prefill speed: %.3f tokens/s",
        prefill_tokens_per_s);
    ET_LOG(
        Info,
        "First token count: 1, extra decode tokens: %" PRId64,
        extra_tokens);
    if (extra_tokens > 0) {
      ET_LOG(Info, "Decode time: %.3f s", decode_ms / 1000.0);
      ET_LOG(Info, "Decode speed: %.3f tokens/s", decode_tokens_per_s);
    } else {
      ET_LOG(
          Info,
          "Decode speed: N/A (need max_new_tokens >= 2 and non-EOS first token)");
    }
    ET_LOG(
        Info,
        "End-to-end generated tokens/s: %.3f",
        end_to_end_tokens_per_s);
    ET_LOG(Info, "=====================");

    return ::executorch::runtime::Error::Ok;
  }
};

std::unique_ptr<InstrumentedMultimodalRunner> create_instrumented_runner(
    const std::string& model_path,
    std::unique_ptr<::tokenizers::Tokenizer> tokenizer) {
  if (!tokenizer || !tokenizer->is_loaded()) {
    ET_LOG(Error, "Tokenizer is null or not loaded");
    return nullptr;
  }

  auto module = std::make_unique<Module>(model_path, Module::LoadMode::File);

  ET_LOG(Info, "Reading metadata from model");
  auto metadata_result =
      ::executorch::extension::llm::get_llm_metadata(tokenizer.get(), module.get());
  if (metadata_result.error() != ::executorch::runtime::Error::Ok) {
    ET_LOG(Error, "Failed to get metadata from model");
    return nullptr;
  }
  auto metadata = metadata_result.get();

  auto eos_ids = std::make_unique<std::unordered_set<uint64_t>>(
      ::executorch::extension::llm::get_eos_ids(tokenizer.get(), module.get()));

  auto io_manager = std::make_unique<IOManager>(*module);
  auto text_decoder_runner =
      std::make_unique<MultimodalDecoderRunner>(module.get(), io_manager.get());
  auto multimodal_prefiller = std::make_unique<MultimodalPrefiller>(
      module.get(),
      text_decoder_runner.get(),
      tokenizer.get(),
      io_manager.get());

  auto stats = std::make_unique<Stats>();
  auto text_token_generator = std::make_unique<TextTokenGenerator>(
      tokenizer.get(),
      text_decoder_runner.get(),
      metadata.at(kUseKVCache),
      std::move(eos_ids),
      stats.get());

  return std::make_unique<InstrumentedMultimodalRunner>(
      std::move(metadata),
      std::move(tokenizer),
      std::move(module),
      std::move(text_decoder_runner),
      std::move(multimodal_prefiller),
      std::move(io_manager),
      std::move(text_token_generator),
      std::move(stats));
}

} // namespace

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

#if defined(ET_USE_THREADPOOL)
  uint32_t num_performant_cores = FLAGS_cpu_threads == -1
      ? ::executorch::extension::cpuinfo::get_num_performant_cores()
      : static_cast<uint32_t>(FLAGS_cpu_threads);
  ET_LOG(
      Info, "Resetting threadpool with num threads = %d", num_performant_cores);
  if (num_performant_cores > 0) {
    ::executorch::extension::threadpool::get_threadpool()
        ->_unsafe_reset_threadpool(num_performant_cores);
  }
#endif

  std::unique_ptr<::tokenizers::Tokenizer> tokenizer =
      ::executorch::extension::llm::load_tokenizer(FLAGS_tokenizer_path);
  if (tokenizer == nullptr) {
    ET_LOG(
        Error, "Failed to load tokenizer from: %s", FLAGS_tokenizer_path.c_str());
    return 1;
  }

  auto runner = create_instrumented_runner(FLAGS_model_path, std::move(tokenizer));
  if (runner == nullptr) {
    ET_LOG(Error, "Failed to create multimodal runner");
    return 1;
  }

  auto load_error = runner->load();
  if (load_error != ::executorch::runtime::Error::Ok) {
    ET_LOG(Error, "Failed to load multimodal runner");
    return 1;
  }

  std::vector<MultimodalInput> inputs = {
      make_text_input("<|im_start|>User:"),
      load_image_bin(FLAGS_image_bin_path),
      make_text_input(
          FLAGS_prompt + std::string("<end_of_utterance>\nAssistant:")),
  };

  GenerationConfig config;
  config.max_new_tokens = FLAGS_max_new_tokens;
  config.temperature = FLAGS_temperature;

  if (FLAGS_warmup) {
    ET_LOG(Info, "Running warmup...");
    auto warmup_error = runner->benchmark(inputs, config);
    if (warmup_error != ::executorch::runtime::Error::Ok) {
      ET_LOG(Error, "Warmup failed");
      return 1;
    }
    runner->reset();
  }

  auto error = runner->benchmark(inputs, config);
  if (error != ::executorch::runtime::Error::Ok) {
    ET_LOG(Error, "Benchmark run failed");
    return 1;
  }

  ET_LOG(Info, "Benchmark finished successfully");
  return 0;
}
