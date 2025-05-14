data_start=0
data_end=3000

python code/data_processing/data_constructor/get_contradiction_data.py \
    --source_doc storysumm \
    --CUDA_VISIBLE_DEVICES 0 \
    --data_start $data_start \
    --data_end $data_end \
    --contra_doc_path data/contradiciton_data/sotrysumm_contra_doc_swap_v3_${data_start}_${data_end}.jsonl \
    --extract_method content_swap

# python code/data_processing/data_constructor/get_consistent_data.py 

# python code/data_processing/data_constructor/merge_contradiction_data.py

# python code/data_processing/data_constructor/combine_data.py

# python code/data_processing/cot_data_distill/cot_data_distill.py

# python code/data_processing/cot_data_distill/analyze_cot_stat.py