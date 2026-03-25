import argparse
import json
import re
import site
import sys
from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F
from transformers import AutoModelForVision2Seq, AutoProcessor


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
WORKSPACE_ROOT = REPO_ROOT.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))


_LLAMA_DEF_LIB = None
_LLAMA_IMPL_LIB = None

_SMOLVLM_TO_META = {
    "model.text_model.embed_tokens.weight": "tok_embeddings.weight",
    "model.text_model.norm.weight": "norm.weight",
    "model.text_model.layers.{}.self_attn.k_proj.weight": "layers.{}.attention.wk.weight",
    "model.text_model.layers.{}.self_attn.q_proj.weight": "layers.{}.attention.wq.weight",
    "model.text_model.layers.{}.self_attn.v_proj.weight": "layers.{}.attention.wv.weight",
    "model.text_model.layers.{}.self_attn.o_proj.weight": "layers.{}.attention.wo.weight",
    "model.text_model.layers.{}.input_layernorm.weight": "layers.{}.attention_norm.weight",
    "model.text_model.layers.{}.post_attention_layernorm.weight": "layers.{}.ffn_norm.weight",
    "model.text_model.layers.{}.mlp.gate_proj.weight": "layers.{}.feed_forward.w1.weight",
    "model.text_model.layers.{}.mlp.down_proj.weight": "layers.{}.feed_forward.w2.weight",
    "model.text_model.layers.{}.mlp.up_proj.weight": "layers.{}.feed_forward.w3.weight",
}


def patch_leafspec_bug() -> None:
    import torch.utils._pytree as pytree

    leaf = pytree.treespec_leaf()
    defaults = {
        "type": None,
        "_context": None,
        "_children": [],
    }
    for name, value in defaults.items():
        if not hasattr(leaf, name):
            object.__setattr__(leaf, name, value)


def _repeat_kv_for_gqa(tensor: torch.Tensor, target_heads: int) -> torch.Tensor:
    current_heads = tensor.size(2)
    if current_heads == target_heads:
        return tensor
    if target_heads % current_heads != 0:
        raise RuntimeError(
            f"Cannot expand KV heads from {current_heads} to {target_heads}"
        )
    return tensor.repeat_interleave(target_heads // current_heads, dim=2)


def _custom_sdpa_impl(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask=None,
    is_causal: bool = False,
    scale=None,
) -> torch.Tensor:
    key = _repeat_kv_for_gqa(key, query.size(2))
    value = _repeat_kv_for_gqa(value, query.size(2))

    q = query.transpose(1, 2)
    k = key.transpose(1, 2)
    v = value.transpose(1, 2)

    kwargs = {
        "attn_mask": attn_mask,
        "dropout_p": 0.0,
        "is_causal": is_causal,
    }
    if scale is not None:
        kwargs["scale"] = float(scale)
    out = F.scaled_dot_product_attention(q, k, v, **kwargs)
    return out.transpose(1, 2).contiguous()


def register_llama_op_shims() -> None:
    global _LLAMA_DEF_LIB, _LLAMA_IMPL_LIB
    if _LLAMA_DEF_LIB is not None:
        return

    _LLAMA_DEF_LIB = torch.library.Library("llama", "DEF")
    _LLAMA_DEF_LIB.define(
        "sdpa_with_kv_cache(Tensor query, Tensor key, Tensor value, Tensor(a!) key_cache, Tensor(b!) value_cache, SymInt start_pos, SymInt seq_len, Tensor? attn_mask=None, float drpout_p=0.0, bool is_causal=False, float? scale=None) -> Tensor"
    )
    _LLAMA_DEF_LIB.define(
        "custom_sdpa(Tensor query, Tensor key, Tensor value, SymInt start_pos, Tensor? attn_mask=None, float drpout_p=0.0, bool is_causal=False, float? scale=None) -> Tensor"
    )
    _LLAMA_DEF_LIB.define(
        "update_cache(Tensor value, Tensor(a!) cache, SymInt start_pos) -> Tensor"
    )
    _LLAMA_DEF_LIB.define(
        "update_cache_with_indices(Tensor value, Tensor(a!) cache, SymInt start_pos, Tensor indices) -> Tensor"
    )
    _LLAMA_DEF_LIB.define("fast_hadamard_transform(Tensor mat) -> Tensor")
    _LLAMA_DEF_LIB.define(
        "custom_quantized_sdpa(Tensor query, Tensor key, Tensor value, SymInt start_pos, Tensor? attn_mask=None, float drpout_p=0.0, bool is_causal=False, float? scale=None, Tensor? q_zero_points=None, Tensor? q_scales=None, Tensor? k_zero_points=None, Tensor? k_scales=None, Tensor? v_zero_points=None, Tensor? v_scales=None, bool is_seq_at_dim_2=False) -> Tensor"
    )

    _LLAMA_IMPL_LIB = torch.library.Library(
        "llama", "IMPL", "CompositeExplicitAutograd"
    )

    def update_cache(
        value: torch.Tensor, cache: torch.Tensor, start_pos
    ) -> torch.Tensor:
        start_pos = int(start_pos)
        seq_len = value.size(1)
        cache[:, start_pos : start_pos + seq_len, :, :].copy_(value)
        return torch.empty((1,), device=cache.device)

    def update_cache_with_indices(
        value: torch.Tensor, cache: torch.Tensor, start_pos, indices: torch.Tensor
    ) -> torch.Tensor:
        del start_pos
        for batch_idx in range(indices.size(0)):
            for seq_idx in range(indices.size(1)):
                index = int(indices[batch_idx, seq_idx].item())
                cache[batch_idx, index].copy_(value[batch_idx, seq_idx])
        return torch.empty((1,), device=cache.device)

    def custom_sdpa(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        start_pos,
        attn_mask=None,
        drpout_p: float = 0.0,
        is_causal: bool = False,
        scale=None,
    ) -> torch.Tensor:
        del start_pos, drpout_p
        return _custom_sdpa_impl(query, key, value, attn_mask, is_causal, scale)

    def sdpa_with_kv_cache(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        start_pos,
        seq_len,
        attn_mask=None,
        drpout_p: float = 0.0,
        is_causal: bool = False,
        scale=None,
    ) -> torch.Tensor:
        del seq_len, drpout_p
        update_cache(key, key_cache, start_pos)
        update_cache(value, value_cache, start_pos)
        return _custom_sdpa_impl(
            query, key_cache, value_cache, attn_mask, is_causal, scale
        )

    def custom_quantized_sdpa(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        start_pos,
        attn_mask=None,
        drpout_p: float = 0.0,
        is_causal: bool = False,
        scale=None,
        q_zero_points=None,
        q_scales=None,
        k_zero_points=None,
        k_scales=None,
        v_zero_points=None,
        v_scales=None,
        is_seq_at_dim_2: bool = False,
    ) -> torch.Tensor:
        del (
            start_pos,
            drpout_p,
            q_zero_points,
            q_scales,
            k_zero_points,
            k_scales,
            v_zero_points,
            v_scales,
            is_seq_at_dim_2,
        )
        return _custom_sdpa_impl(
            query.to(torch.float32),
            key.to(torch.float32),
            value.to(torch.float32),
            attn_mask,
            is_causal,
            scale,
        )

    _LLAMA_IMPL_LIB.impl("update_cache", update_cache)
    _LLAMA_IMPL_LIB.impl("update_cache_with_indices", update_cache_with_indices)
    _LLAMA_IMPL_LIB.impl("custom_sdpa", custom_sdpa)
    _LLAMA_IMPL_LIB.impl("sdpa_with_kv_cache", sdpa_with_kv_cache)
    _LLAMA_IMPL_LIB.impl("fast_hadamard_transform", lambda mat: mat)
    _LLAMA_IMPL_LIB.impl("custom_quantized_sdpa", custom_quantized_sdpa)


def _map_key(key: str) -> str:
    for old_pattern, replacement in _SMOLVLM_TO_META.items():
        mapped = re.sub(
            old_pattern.replace("{}", r"([0-9]+)"),
            replacement.replace("{}", r"\1"),
            key,
        )
        if mapped != key:
            return mapped
    raise KeyError(key)


def convert_text_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    converted = {}
    for key, value in state_dict.items():
        try:
            converted[_map_key(key)] = value
        except KeyError:
            continue
    converted["output.weight"] = state_dict["lm_head.weight"]
    return converted


def _tuple_output(value):
    if isinstance(value, tuple):
        return value
    return (value,)


class SmolVLMVisionEncoder(torch.nn.Module):
    def __init__(self, hf_model):
        super().__init__()
        vision_model = hf_model.model.vision_model
        self.patch_embedding = vision_model.embeddings.patch_embedding
        self.position_embedding = vision_model.embeddings.position_embedding
        self.encoder = vision_model.encoder
        self.post_layernorm = vision_model.post_layernorm
        self.connector = hf_model.model.connector
        patches_per_side = vision_model.embeddings.num_patches_per_side
        num_patches = patches_per_side * patches_per_side
        self.register_buffer(
            "position_ids",
            torch.arange(num_patches, dtype=torch.long).unsqueeze(0),
            persistent=False,
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        batch_size = pixel_values.size(0)
        hidden_states = self.patch_embedding(pixel_values).flatten(2).transpose(1, 2)
        hidden_states = hidden_states + self.position_embedding(
            self.position_ids.expand(batch_size, -1)
        )
        encoder_outputs = self.encoder(inputs_embeds=hidden_states, attention_mask=None)
        last_hidden_state = self.post_layernorm(encoder_outputs.last_hidden_state)
        return _tuple_output(self.connector(last_hidden_state))


class SmolVLMTokenEmbedding(torch.nn.Module):
    def __init__(self, hf_model):
        super().__init__()
        self.embed_tokens = hf_model.model.text_model.embed_tokens

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return _tuple_output(self.embed_tokens(tokens))


class SmolVLMTextDecoder(torch.nn.Module):
    def __init__(self, text_model):
        super().__init__()
        self.text_model = text_model

    def forward(self, embeddings: torch.Tensor, input_pos: torch.Tensor) -> torch.Tensor:
        return _tuple_output(self.text_model(None, {"input_pos": input_pos}, embeddings))


def build_text_model(
    text_state_dict: Dict[str, torch.Tensor], max_seq_len: int, max_context_len: int
):
    ensure_llama_custom_ops_loaded()
    from executorch.extension.llm.custom_ops import custom_ops  # noqa: F401

    from executorch.examples.models.llama.llama_transformer import construct_transformer
    from executorch.examples.models.llama.model_args import ModelArgs
    from executorch.examples.models.llama.source_transformation.custom_kv_cache import (
        replace_kv_cache_with_custom_kv_cache,
    )
    from executorch.examples.models.llama.source_transformation.sdpa import (
        replace_sdpa_with_custom_op,
    )

    params_path = REPO_ROOT / "examples/models/smolvlm/500M_config.json"
    with open(params_path, "r") as f:
        params = json.load(f)

    model_args = ModelArgs(
        max_seq_len=max_seq_len,
        max_context_len=max_context_len,
        max_batch_size=1,
        use_kv_cache=True,
        use_sdpa_with_kv_cache_op=True,
        enable_dynamic_shape=True,
        **params,
    )

    with torch.device("meta"):
        model = construct_transformer(model_args)
        model = replace_kv_cache_with_custom_kv_cache(model)
        model = replace_sdpa_with_custom_op(model)
        model.load_state_dict(text_state_dict, strict=False, assign=True)

    return model.eval()


def _unwrap_custom_sdpa_cache_arg(arg):
    if not isinstance(arg, torch.fx.Node):
        return None
    node = arg
    while isinstance(node, torch.fx.Node):
        target = str(node.target)
        if target in {"aten.to.dtype", "aten.contiguous.default"}:
            first_arg = node.args[0]
            if not isinstance(first_arg, torch.fx.Node):
                return None
            node = first_arg
            continue
        if target == "aten.transpose.int":
            base = node.args[0]
            return base if isinstance(base, torch.fx.Node) else None
        break
    return None


def normalize_custom_sdpa_cache_args(exported_program):
    # The CPU export path already produces K/V cache tensors in the
    # [B, S, H, D] layout expected by llama.custom_sdpa. On this model the
    # graph contains a transpose(transpose(cache)) pair before a dtype cast;
    # collapsing only part of that chain corrupts the layout and causes the
    # runtime SDPA kernel to interpret sequence length as KV head count.
    return exported_program


def export_modules(args):
    patch_leafspec_bug()
    ensure_llama_custom_ops_loaded()
    from executorch.extension.llm.custom_ops import custom_ops  # noqa: F401

    model = AutoModelForVision2Seq.from_pretrained(
        args.model_dir,
        local_files_only=True,
        device_map="cpu",
        torch_dtype=torch.float32,
    ).eval()
    processor = AutoProcessor.from_pretrained(args.model_dir, local_files_only=True)

    from executorch.exir import (
        EdgeCompileConfig,
        ExecutorchBackendConfig,
        to_edge_transform_and_lower,
    )
    from executorch.exir.passes import MemoryPlanningPass

    text_state_dict = convert_text_state_dict(model.state_dict())
    text_model = build_text_model(text_state_dict, args.max_seq_len, args.max_context_len)

    vision_encoder = SmolVLMVisionEncoder(model).eval()
    token_embedding = SmolVLMTokenEmbedding(model).eval()
    text_decoder = SmolVLMTextDecoder(text_model).eval()

    image_input = (torch.zeros(1, 3, 512, 512, dtype=torch.float32),)
    token_input = (torch.arange(1, args.max_context_len + 1, dtype=torch.long).unsqueeze(0),)
    embedding_input = (
        torch.zeros(1, args.max_context_len, 960, dtype=torch.float32),
        torch.tensor([0], dtype=torch.long),
    )

    token_dim = torch.export.Dim("token_dim", min=1, max=args.max_context_len)
    token_shapes = ({1: token_dim},)
    embedding_shapes = ({1: token_dim}, {0: 1})

    with torch.no_grad():
        vision_ep = torch.export.export(vision_encoder, image_input, strict=False)
        token_ep = torch.export.export(
            token_embedding,
            token_input,
            dynamic_shapes=token_shapes,
            strict=False,
        )
        text_ep = torch.export.export(
            text_decoder,
            embedding_input,
            dynamic_shapes=embedding_shapes,
            strict=False,
        )
        text_ep = normalize_custom_sdpa_cache_args(text_ep)

    metadata = {
        "get_max_seq_len": args.max_seq_len,
        "get_max_context_len": args.max_context_len,
        "get_vocab_size": len(processor.tokenizer),
        "get_bos_id": processor.tokenizer.bos_token_id,
        "get_eos_ids": [processor.tokenizer.eos_token_id],
        "use_kv_cache": True,
        "use_sdpa_with_kv_cache": True,
        "enable_dynamic_shape": True,
    }

    edge = to_edge_transform_and_lower(
        {
            "vision_encoder": vision_ep,
            "token_embedding": token_ep,
            "text_decoder": text_ep,
        },
        partitioner=[],
        compile_config=EdgeCompileConfig(
            _check_ir_validity=False,
            _skip_dim_order=True,
        ),
        constant_methods=metadata,
    )

    return edge.to_executorch(
        config=ExecutorchBackendConfig(
            extract_delegate_segments=True,
            do_quant_fusion_and_const_prop=True,
            memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False),
        )
    )


def ensure_llama_custom_ops_loaded() -> None:
    try:
        _ = torch.ops.llama.custom_sdpa.out
        _ = torch.ops.llama.update_cache.out
        return
    except (AttributeError, RuntimeError):
        pass

    from executorch.extension.pybindings import portable_lib  # noqa: F401

    for base in site.getsitepackages():
        candidate = (
            Path(base)
            / "executorch/extension/llm/custom_ops/libcustom_ops_aot_lib.so"
        )
        if candidate.exists():
            torch.ops.load_library(candidate)
            return

    raise FileNotFoundError("libcustom_ops_aot_lib.so not found in site-packages")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--max-context-len", type=int, default=256)
    args = parser.parse_args()

    program = export_modules(args)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        program.write_to_file(f)
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
