basic information
1. my current computer is ubuntu22.04, my root password is 1, you can use it for any root operation to realize my goal
2. my computer is connecting with an android phone (oneplus ace5 equipped with snapdragon8 gen3) through adb, you will see it by using adb devices
3. my conde env is executorch-XNNPACK; my android ndk path is /home/zzl/project/AndroidNDK26;

my goal
1. you should run smolvlm on my phone(with cpu only), model path is /home/zzl/.cache/huggingface/hub/models--HuggingFaceTB--SmolVLM-500M-Instruct/snapshots/a7da5b986cb59b408707209984f360a5f4ad7e47
2. to run smolvlm on phone using cpu, you should export smolvlm model to pte file, then push the file and related library and executable file to my phone, executing them finally.

some suggestion
1. you can search toturial under /home/zzl/project4/execu-XNNPACK/executorch/docs to know how to export and run a model on device using cpu. but the suggestion is not full, you may need to expore more tutorials under /home/zzl/project4/execu-VULKAN/executorch.
2. you can follow /home/zzl/project4/executorch/exp_scripts/new_run.sh to know how to run smolvlm on device using npu, this is a example, you do not need to run the instruction, I want you run smolvlm on device using cpu only.

your task
1. please finish my goal with at all costs, your all codes should be under /home/zzl/project4/execu-XNNPACK/executorch/smolvlm_cpu
2. summarize your all operations to note.md, analyse the final experiment results



