deepspeed --include=localhost:4,5,6,7 \
          --master_addr=127.0.0.1 \
          --master_port=29502 \
          code/rl/rl.py \
          --model_name your_sft_model_path \
          --output_dir output/ \
          --output_name llama3_format \
          --data_path data/ContraDocX_rl_raw.json
