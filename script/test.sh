
# base
python code/eval/eval_metric.py \
    --data_name ContraDoc.json \
    --devices 1 \
    --np_save_dir npz/llama_base.npz \
    --model /data2/LLM_Model/Meta-Llama-3.1-8B-Instruct

# python code/eval/eval_metric.py \
#     --data_name ContraDoc.json \
#     --devices 1 \
#     --np_save_dir npz/qwen_base.npz \
#     --model /data2/LLM_Model/Qwen2.5-7B-Instruct


# cot
# python code/eval/eval_metric.py \
#     --data_name ContraDoc.json \
#     --devices 1 \
#     --CoT \
#     --np_save_dir npz/llama_cot.npz \
#     --model /data2/LLM_Model/Meta-Llama-3.1-8B-Instruct

# python code/eval/eval_metric.py \
#     --data_name ContraDoc.json \
#     --devices 1 \
#     --CoT \
#     --np_save_dir npz/qwen_cot.npz \
#     --model /data2/LLM_Model/Qwen2.5-7B-Instruct

# sft
# python code/eval/eval_metric.py \
#     --data_name ContraDoc.json \
#     --devices 1 \
#     --R1 \
#     --np_save_dir npz/llama_sft1.npz \
#     --model your_sft_model_path (llama or qwen)

# rl
# python code/eval/eval_metric.py \
#     --data_name ContraDoc.json \
#     --devices 1 \
#     --R1 \
#     --lora \
#     --lora_path your_lora_path \
#     --np_save_dir npz/qwen_rl_cover.npz \
#     --model your_sft_model_path (llama or qwen)
