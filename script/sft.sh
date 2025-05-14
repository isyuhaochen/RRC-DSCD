deepspeed --include=localhost:1 --master_addr=127.0.0.1 --master_port=29504 code/sft/sft.py \
    --model_path /data2/LLM_Model/Meta-Llama-3.1-8B-Instruct \
    --output_dir output/ \
    --output_name llama3_sft_test \
    --data_path data/distill_cot_sft.json