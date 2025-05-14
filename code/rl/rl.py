import re
import json
import random
import argparse
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer
from rewards import get_reward_funcs
from datasets import Dataset
from nltk.tokenize import sent_tokenize
from peft import get_peft_model, LoraConfig, TaskType


def build_prompt_and_evidence_labels(text, evidences, tokenizer, model_name):
    sentences = sent_tokenize(text)
    formatted_text = " ".join([f"[{i + 1}] {sent}" for i, sent in enumerate(sentences)])

    prompt = (
        "The task is to determine whether the article contains any self-contradictions. "
        "If yes, provide evidence by quoting mutually contradictory sentences in a list of strings in Python. "
        "If no, give an empty list.\n\n"
        f"Article: {formatted_text}\n\n"
        "Response: Form your answer in the following format (OR options are provided). "
        "Please answer the Judgment and Evidence in the prescribed format. "
        "Evidence must be a list that can be parsed by Python:\n\n"
        "Judgment: yes OR no\n"
        "Evidence: [\"xxxxxx\", \"xxxxxx\", ..., \"xxxxxx\"] OR []"
    )

    if 'qwen' in model_name:
        messages = [{"role": "user", "content": prompt}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt = (
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )

    label_indices = []
    for evidence in evidences:
        for idx, sentence in enumerate(sentences):
            if evidence.lower() in sentence.lower() or sentence.lower() in evidence.lower():
                label_indices.append(idx + 1)
                break

    return prompt, label_indices, len(sentences)


def load_dataset(data_path="data/ContraDocX_rl_raw.json", tokenizer=None, model_name=None):
    with open(data_path) as f:
        raw_data = json.load(f)

    dataset = []
    for label_key in ['pos', 'neg']:
        for item in raw_data[label_key].values():
            evidences = item.get('contra_list', [])
            text = item['text']

            prompt, evidence_labels, sentence_count = build_prompt_and_evidence_labels(
                text, evidences, tokenizer, model_name
            )

            dataset.append({
                'prompt': prompt,
                'label': label_key == 'pos',
                'evidence': evidences,
                'evidence_label_list': evidence_labels,
                'doc_sentence_num': sentence_count
            })

    random.seed(42)
    random.shuffle(dataset)
    return dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='your_model_name')
    parser.add_argument('--output_dir', type=str, default='output/')
    parser.add_argument('--output_name', type=str, default='llama3_format')
    parser.add_argument('--data_path', type=str, default='data/ContraDocX_rl_raw.json')
    parser.add_argument('--local_rank', type=int, default=-1,
                    help='Local rank passed by deepspeed') 
    return parser.parse_args()


def main():
    args = parse_args()

    training_args = GRPOConfig(
        output_dir=f"{args.output_dir}{args.output_name}",
        learning_rate=5e-5,
        logging_steps=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        num_generations=4,
        max_prompt_length=8192,
        max_completion_length=4096,
        num_train_epochs=1,
        save_steps=100,
        deepspeed="code/rl/ds_config.json",
        gradient_checkpointing=True,
        bf16=True,
        logging_dir=f"logs/{args.output_name}",
        report_to="tensorboard",
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map=None
    ).to("cuda")

    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        task_type=TaskType.CAUSAL_LM
    )

    model.enable_input_require_grads()
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.config.use_cache = False

    data = load_dataset(args.data_path, tokenizer=tokenizer, model_name=args.model_name)
    dataset = Dataset.from_list(data)

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=get_reward_funcs(),
        args=training_args,
        train_dataset=dataset,
    )

    trainer.train()

    # Save LoRA weights and tokenizer
    lora_save_path = f"lora/{args.output_name}"
    trainer.model.save_pretrained(lora_save_path)
    tokenizer.save_pretrained(lora_save_path)


if __name__ == "__main__":
    main()