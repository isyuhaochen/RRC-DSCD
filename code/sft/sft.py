import os
import json
import torch
import random
import argparse
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    TrainingArguments, Trainer,
    DataCollatorForSeq2Seq
)
from peft import get_peft_model, LoraConfig, TaskType
from datasets import Dataset
from nltk.tokenize import sent_tokenize


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a language model with LoRA.")
    parser.add_argument('--model_path', type=str, default='model',
                        help='Path to the base model')
    parser.add_argument('--output_dir', type=str, default='output/',
                        help='Base output directory for logs and checkpoints')
    parser.add_argument('--output_name', type=str, default='llama3_sft',
                        help='Name of the specific fine-tuning run')
    parser.add_argument('--data_path', type=str, default='data/ContraDocX_sft.json',
                        help='Path to the training data')
    parser.add_argument('--local_rank', type=int, default=-1,
                    help='Local rank passed by deepspeed') 
                   
    return parser.parse_args()


def load_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="cuda"
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    print(f"Pad token ID: {tokenizer.pad_token_id}")

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

    return model, tokenizer


def format_example(example, tokenizer, model_name='llama'):
    """Format a single training example into input-output pairs for fine-tuning."""
    prompt = (
        "The task is to determine whether the article contains any self-contradictions. "
        "If yes, provide evidence by quoting mutually contradictory sentences in a list of strings in Python. "
        "If no, give an empty list.\n\n"
        "Article: "
        f"{example['input']}\n\n"
        "Response: Form your answer in the following format (OR options are provided). "
        "Please answer the Judgment and Evidence in the prescribed format. "
        "Evidence must be a list that can be parsed by Python:\n\n"
        "Judgment: yes OR no\n"
        "Evidence: [\"xxxxxx\", \"xxxxxx\", ..., \"xxxxxx\"] OR []"
    )

    if 'qwen' in model_name.lower():
        response = tokenizer(example['output'] + tokenizer.eos_token, add_special_tokens=False, max_length=8192)
        messages = [{"role": "user", "content": prompt}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt = (
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            + prompt +
            "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        response = tokenizer(f"{example['output']}<|eot_id|>", add_special_tokens=False, max_length=8192)

    instruction = tokenizer(prompt, add_special_tokens=False, max_length=8192)

    input_ids = instruction["input_ids"] + response["input_ids"]
    attention_mask = instruction["attention_mask"] + response["attention_mask"]
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"]

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long)
    }


def load_dataset(data_path):
    """Load and format dataset from JSON file."""
    with open(data_path, "r") as f:
        raw_data = json.load(f)

    examples = []

    def add_entries(entries, max_count=3000):
        count = 0
        for item in entries.values():
            sentences = sent_tokenize(item["text"])
            formatted_text = " ".join([f"[{i + 1}] {sent}" for i, sent in enumerate(sentences)])
            output = f"<think>\n{item['reasoning_content']}\n</think>\n<answer>\n{item['content']}\n</answer>"
            examples.append({"input": formatted_text, "output": output})
            count += 1
            if count >= max_count:
                break

    add_entries(raw_data.get("pos", {}))
    add_entries(raw_data.get("neg", {}))

    random.shuffle(examples)
    print(f"Total examples loaded: {len(examples)}")
    return Dataset.from_list(examples)


def main():
    args = parse_args()
    model, tokenizer = load_model_and_tokenizer(args.model_path)
    dataset = load_dataset(args.data_path)

    # Apply formatting
    tokenized_dataset = dataset.map(lambda x: format_example(x, tokenizer, args.model_path))

    # Training configuration
    training_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, args.output_name),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=1e-4,
        num_train_epochs=3,
        save_steps=200,
        logging_steps=10,
        logging_dir=os.path.join("logs", args.output_name),
        report_to="tensorboard",
        deepspeed="code/sft/ds_config.json",
        gradient_checkpointing=True,
        bf16=True
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        pad_to_multiple_of=8
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator
    )

    trainer.train()

    # Save final model and tokenizer
    final_model_path = os.path.join(args.output_dir, args.output_name, "final")
    trainer.model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    print(f"Model and tokenizer saved to {final_model_path}")


if __name__ == "__main__":
    main()
