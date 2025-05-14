import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Set the visible GPU device (e.g., only GPU 6 will be used)
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

# Define model paths
base_model_path = "/data2/LLM_Model/Qwen2.5-7B-Instruct"  # e.g., "meta-llama/Llama-2-7b-hf"
lora_model_path = "sft_lora/qwen_sft_3/final"  # Path to the LoRA adapter

# Load the base model (automatically assigned to available devices)
model = AutoModelForCausalLM.from_pretrained(base_model_path, device_map="auto")

# Load the LoRA adapter
model = PeftModel.from_pretrained(model, model_id=lora_model_path, adapter_name="default")

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_path)

# Merge LoRA weights into the base model and unload adapters
model = model.merge_and_unload()

# Define the path to save the merged model
merged_model_path = "model_weight/qwen_sft_3"

# Save the merged model and tokenizer
model.save_pretrained(merged_model_path)
tokenizer.save_pretrained(merged_model_path)
