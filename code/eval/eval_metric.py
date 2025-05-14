import os
import re
import ast
import json
import torch
import argparse
import logging
import numpy as np
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "../"))

from nltk.tokenize import sent_tokenize
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm
from evaluate import load
from model import get_model_response

# Set up logging
logging.basicConfig(level=logging.INFO)

# Prompt configurations
def get_prompt_templates():
    return {
        "base": {
            "beginning": """The task is to determine whether the article contains any self-contradictions. If yes, provide evidence by quoting mutually contradictory sentences in a list of strings in Python. If no, give an empty list.\n\nArticle:  """,
            "end": """
Response: Form your answer in the following format (OR options are provided). Please answer the Judgment and Evidence in the prescribed format. Evidence must be a list that can be parsed by Python:

Judgment: yes OR no
Evidence: ["xxxxxx", "xxxxxx", ..., "xxxxxx"] OR []
""",
            "pos_only": False,
            "extract": extract_info
        },
        "cot": {
            "beginning": """The task is to determine whether the article contains any self-contradictions. If yes, provide evidence by quoting mutually contradictory sentences in a list of strings in Python. If no, give an empty list.\n\nArticle:  """,
            "end": """
Response: Form your answer in the following format (OR options are provided). Please answer the Judgment and Evidence in the prescribed format. Evidence must be a list that can be parsed by Python:

Judgment: yes OR no
Evidence: ["xxxxxx", "xxxxxx", ..., "xxxxxx"] OR []
Please think step by step, and finally give the answer.
""",
            "pos_only": False,
            "extract": extract_info
        },
        "r1": {
            "beginning": """The task is to determine whether the article contains any self-contradictions. If yes, provide evidence by quoting mutually contradictory sentences in a list of strings in Python. If no, give an empty list.\n\nArticle:  """,
            "end": """
Response: Form your answer in the following format (OR options are provided). Please answer the Judgment and Evidence in the prescribed format. Evidence must be a list that can be parsed by Python:

Judgment: yes OR no
Evidence: ["xxxxxx", "xxxxxx", ..., "xxxxxx"] OR []
""",
            "pos_only": False,
            "extract": extract_info
        }
    }


def get_model_ans_local(prompt, model, tokenizer, model_name, max_new_tokens=4096):
    """Generate response using a local model."""
    if 'qwen' in model_name.lower():
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
        return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    prompt_formatted = f'<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
    inputs = tokenizer(prompt_formatted, add_special_tokens=False, return_tensors="pt").to(model.device)
    tokenizer.pad_token = tokenizer.eos_token
    output = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.batch_decode(output[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()


def load_docs_with_evidence(path, pos_only=False):
    """Load documents and associated evidence from JSON."""
    with open(path, 'r') as f:
        data = json.load(f)

    docs, evidences, info, refs = [], [], [], []
    positive_data = data["pos"]
    logging.info(f"Loaded {len(positive_data)} positive examples.")

    for key, value in positive_data.items():
        item = value.copy()
        item.update({"label": True, "id": key})
        info.append(item)
        docs.append(item["text"])
        evidences.append(item.get("evidence", item.get("contra_list", [])))
        refs.append(item.get("ref", 'None'))

    if not pos_only:
        negative_data = data["neg"]
        logging.info(f"Loaded {len(negative_data)} negative examples.")
        for key, value in negative_data.items():
            item = value.copy()
            item.update({"label": False, "evidence": None, "id": key})
            info.append(item)
            docs.append(item["text"])
            evidences.append(None)
            refs.append(item.get("ref", 'None'))

    assert len(docs) == len(evidences) == len(info) == len(refs)
    return docs, evidences, info, refs


def load_prompted_docs(docs, bs_prompt, es_prompt):
    """Wrap each document with its beginning and end prompt."""
    return [bs_prompt + (doc if doc.endswith("\n") else doc + "\n") + es_prompt for doc in docs]


def convert_string_to_list(input_str: str):
    """Convert string to Python list if it's a valid list literal."""
    match = re.search(r'\[.*?\]', input_str.replace("\n", "").strip())
    if not match:
        raise ValueError("No list found in string.")
    return ast.literal_eval(match.group(0))


def extract_info(full_response: str):
    """Extract judgment and evidence from model response."""
    info = {"response": full_response}
    lower_response = full_response.lower()
    start = lower_response.find("\njudgment") + len("\njudgment")
    end = lower_response.find("\n", start)
    judge = lower_response[start:end].strip()

    if "yes" in judge:
        info["judgment"] = "yes"
        try:
            info["evidence"] = convert_string_to_list(full_response.split("evidence:")[1])
        except Exception:
            info["evidence"] = []
            logging.warning("Judged 'yes' but failed to extract evidence.")
    else:
        info["judgment"] = "no"
        info["evidence"] = []

    return info


def calculate_metrics(gold_labels, predicted_labels, full_info):
    """Calculate classification metrics."""
    TP = FP = TN = FN = 0
    stats = {"TP": [], "FP": [], "TN": [], "FN": []}

    for gold, pred, meta in zip(gold_labels, predicted_labels, full_info):
        if gold and pred:
            TP += 1
            stats["TP"].append(meta)
        elif not gold and not pred:
            TN += 1
            stats["TN"].append(meta)
        elif gold and not pred:
            FN += 1
            stats["FN"].append(meta)
        elif not gold and pred:
            FP += 1
            stats["FP"].append(meta)

    precision = TP / (TP + FP) if TP + FP else 0
    recall = TP / (TP + FN) if TP + FN else 0
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0

    return {
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "Accuracy": accuracy,
        "True Positive": TP,
        "False Positive": FP,
        "True Negative": TN,
        "False Negative": FN
    }

def sentence_cover(completion, total_num):
    def parse_ranges(range_str):
        numbers = set()
        matches = re.findall(r'\[(\d+)(?:-(\d+))?\]', range_str)

        for match in matches:
            start = int(match[0])
            end = int(match[1]) if match[1] else start  
            numbers.update(range(start, end + 1))
        return numbers
    try:
        reasoning_content = completion.split("<think>")[1].split("</think>")[0].lower()
        length = len(parse_ranges(reasoning_content))
        return length / total_num 
    except Exception as e:
        return 0

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Self-Contradiction Detection')
    parser.add_argument('--data_path', type=str, default='data/')
    parser.add_argument('--data_name', type=str, default='ContraDocX_test.json')
    parser.add_argument('--CoT', action='store_true')
    parser.add_argument('--R1', action='store_true')
    parser.add_argument('--save_ans', action='store_true')
    parser.add_argument('--save_path', type=str, default='sft_data_create/ContraDocX_train_with_ref.json')
    parser.add_argument('--np_save_dir', type=str, default='predictions.npz')
    parser.add_argument('--devices', type=str, default='0')
    parser.add_argument('--online_model', action='store_true')
    parser.add_argument('--lora', action='store_true')
    parser.add_argument('--lora_path', type=str, default='model_weight/checkpoint-800')
    parser.add_argument('--model', type=str, default='/data2/LLM_Model/Meta-Llama-3.1-8B-Instruct')
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.devices
    logging.info(f"Using CUDA devices: {args.devices}")

    if args.online_model:
        model, tokenizer = None, None
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto")
        if args.lora:
            model = PeftModel.from_pretrained(model, args.lora_path)

    bertscore = load("bertscore", device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    logging.info("Loaded BERTScore.")

    prompts_config = get_prompt_templates()
    prompt_key = "base"
    if args.CoT:
        prompt_key = "cot"
    elif args.R1:
        prompt_key = "r1"

    prompt_template = prompts_config[prompt_key]
    docs, evidences, info, refs = load_docs_with_evidence(os.path.join(args.data_path, args.data_name), pos_only=prompt_template["pos_only"])

    
    # Format documents with indexed sentences
    sentences_num = []
    for i in range(len(docs)):
        sentences = sent_tokenize(docs[i])
        docs[i] = " ".join([f"[{index + 1}] {sent}" for index, sent in enumerate(sentences)])
        sentences_num.append(len(sentences))
    prompts = load_prompted_docs(docs, prompt_template["beginning"], prompt_template["end"])

    responses = []
    
    for i, prompt in tqdm(enumerate(prompts), total=len(prompts), desc="Generating Responses"):
        logging.info(f"Processing document ID: {info[i]['id']}")
        response = get_model_response(prompt, model=args.model) if args.online_model else get_model_ans_local(prompt, model, tokenizer, args.model)
        responses.append(response)
        
    assert len(prompts) == len(evidences) == len(info) == len(responses)
    char_counts = [len(resp) for resp in responses]
    word_counts = [len(resp.split()) for resp in responses]

    sentence_cover_list = [sentence_cover(responses[i],sentences_num[i]) for i in range(len(responses))] 
    
    verified_predictions, non_verified_predictions = [], []
    save_data = {}
    for i in range(len(prompts)):
        response_info = prompt_template["extract"](responses[i].strip())
        non_verified_predictions.append(True if response_info["judgment"]=="yes" else False)

        if non_verified_predictions[-1] and info[i]["label"]: # only verify when both true
            verify = yes_verify(response_info["evidence"], evidences[i], bertscore)
            if verify: 
                verified_predictions.append(True)
                if args.save_ans:
                    save_data[info[i]['id']] = {"id":info[i]['id'],"prompt":prompts[i],"evidence":evidences[i],"model_ans":responses[i],"label":info[i]["label"]}
            else: 
                verified_predictions.append(False)
        else: 
            if not non_verified_predictions[-1] and not info[i]["label"] and args.save_ans:
                save_data[info[i]['id']] = {"id":info[i]['id'],"prompt":prompts[i],"evidence":evidences[i],"model_ans":responses[i],"label":info[i]["label"]}
                
            verified_predictions.append(non_verified_predictions[-1])

    if args.save_ans:
        with open(args.save_path,'w') as f:
            json.dump(save_data,f,indent=4)

    without_verify_res = calculate_metrics([x["label"] for x in info],
                                       non_verified_predictions, info)
    verify_res = calculate_metrics([x["label"] for x in info], verified_predictions, info)


    np.savez(args.np_save_dir, non_verified=non_verified_predictions, verified=verified_predictions,char_counts=char_counts,word_counts=word_counts,sentence_cover_list=sentence_cover_list)
    print(f"without_verify: {without_verify_res}")
    print(f"verify: {verify_res}") 

if __name__ == "__main__":
    main()