import argparse
import os
import sys
import random
import re
import logging
import json
import torch
import ast
import numpy as np


current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "../../"))

from tqdm import tqdm
from nltk.tokenize import sent_tokenize
from collections import defaultdict
from model import get_model_response_with_reasoning
from evaluate import load

# Set up logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# Set CUDA
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Load evaluation metric
bertscore = load("bertscore", device=torch.device("cuda"))
logging.info("BERTScore loaded successfully.")


def load_docs_evis_infos(path, pos_only=False):
    """
    Load documents and evidence from JSON dataset.
    """
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    docs, evidences, info = [], [], []

    positive_data = data.get("pos", {})
    logging.info(f"Positive examples: {len(positive_data)}")
    for item in positive_data.values():
        item["label"] = True
        docs.append(item["text"])
        evidences.append(item.get("evidence", item.get("contra_list", [])))
        info.append(item)

    if not pos_only:
        negative_data = data.get("neg", {})
        logging.info(f"Negative examples: {len(negative_data)}")
        for item in negative_data.values():
            item["label"] = False
            item["evidence"] = None
            docs.append(item["text"])
            evidences.append(None)
            info.append(item)

    assert len(docs) == len(evidences) == len(info)
    combined = list(zip(docs, evidences, info))
    random.shuffle(combined)
    docs, evidences, info = zip(*combined)
    return list(docs), list(evidences), list(info)


def load_prompted_doc(docs, bs_prompt="", es_prompt=""):
    return [f"{bs_prompt}{doc.strip()}\n{es_prompt}" for doc in docs]


def contains_bracketed_integer(s):
    return bool(re.search(r"\[\d+\]", s))


def contains_key_word(s, prompt_type=1):
    if prompt_type == 1:
        return 'Judgment' in s and 'Evidence' in s
    elif prompt_type == 2:
        return all(f"<Question {i}>" in s and f"<Answer {i}>" in s for i in range(1, 7)) and 'Judgment' in s and 'Evidence' in s
    return False


def extract_info(response):
    info = {"response": response}
    response_lower = response.lower()
    judgment_start = response_lower.find("\njudgment") + len("\njudgment")
    judgment_end = response_lower.find("\n", judgment_start)
    judge = response_lower[judgment_start:judgment_end].strip()

    if "yes" in judge:
        info["judgment"] = "yes"
        try:
            evidence = convert_string_to_list(response.split("Evidence:")[1])
        except Exception:
            evidence = []
        info["evidence"] = evidence
    else:
        info["judgment"] = "no"
        info["evidence"] = []
    return info


def yes_verify(pred_evs, gold_evs):
    if not pred_evs:
        return False

    pred_evs = [ev.lower() for ev in pred_evs]
    gold_evs = [gold_evs.lower()] if isinstance(gold_evs, str) else [ev.lower() for ev in gold_evs]

    for gold in gold_evs:
        if direct_find(pred_evs, gold) is not None:
            return True
        pr, rc = batch_bs(pred_evs, gold)
        if pr > 0.98 or rc > 0.98:
            return True
    return False


def direct_find(sents, query):
    for i, sent in enumerate(sents):
        if sent in query or query in sent:
            return i
    return None


def batch_bs(predictions, reference, batch_size=32):
    pr_scores, rc_scores = [], []

    for i in range(0, len(predictions), batch_size):
        batch = predictions[i:i + batch_size]
        refs = [reference] * len(batch)

        results = bertscore.compute(
            predictions=refs,
            references=batch,
            lang="en",
            model_type="../model/roberta-large",
            num_layers=24,
            device=torch.device("cuda")
        )

        pr_scores.extend(results["precision"])
        rc_scores.extend(results["recall"])

    return max(pr_scores), max(rc_scores)


def convert_string_to_list(s):
    s = s.strip().replace("\n", "")
    match = re.search(r'\[.*\]', s)
    if not match:
        raise ValueError("No valid list found.")
    return ast.literal_eval(match.group(0))


def process_data(data_list):
    return {item['unique_id']: item for item in data_list}


def main():
    prompt_config = {
        "beginning": (
            "The task is to determine whether the article contains any self-contradictions. "
            "If yes, provide evidence by quoting mutually contradictory sentences in a list of strings in Python. "
            "If no, give an empty list.\nArticle:  "
        ),
        "end": (
            "\nIn the <think></think> part, the sentences mentioned need to be numbered [i] at the same time. "
            "In Evidence, the number [i] is not required:\n"
            "Response: Form your answer in the following format (OR options are provided), "
            "Please answer the Judgment and Evidence in the prescribed format. "
            "Evidence must be a list that can be parsed by Python:\n"
            "Judgment: yes OR no\n"
            "Evidence: [\"xxxx\", ...] OR []"
        ),
        "pos_only": False
    }

    input_path = "test_data/merged_rl.json"
    docs, evidences, info = load_docs_evis_infos(input_path, pos_only=prompt_config["pos_only"])
    docs = docs[:5]
    evidences = evidences[:5]
    info = info[:5]

    max_sentences = 0
    for i in range(len(docs)):
        sentences = sent_tokenize(docs[i])
        docs[i] = " ".join(f"[{j + 1}] {sent}" for j, sent in enumerate(sentences))
        max_sentences = max(max_sentences, len(sentences))
    logging.info(f"Max sentence count in docs: {max_sentences}")

    prompts = load_prompted_doc(docs, bs_prompt=prompt_config["beginning"], es_prompt=prompt_config["end"])

    pos_results, neg_results = [], []

    for i in tqdm(range(len(prompts)), desc="Processing Prompts"):
        logging.info(f"Processing sample {i}, ID: {info[i]['unique_id']}")
        content, reasoning_content = get_model_response_with_reasoning(prompts[i])

        if contains_bracketed_integer(reasoning_content) and contains_key_word(content, prompt_type=1):
            result = extract_info(content)
            result["reasoning_content"] = reasoning_content
            result["content"] = content

            if result["judgment"] == "yes" and info[i]["label"]:
                if yes_verify(result["evidence"], evidences[i]):
                    pos_results.append({**info[i], **result})
            elif result["judgment"] == "no" and not info[i]["label"]:
                neg_results.append({**info[i], **result})

    logging.info(f"Valid positive examples: {len(pos_results)}")
    logging.info(f"Valid negative examples: {len(neg_results)}")

    final_data = {'pos': process_data(pos_results), 'neg': process_data(neg_results)}
    output_file = 'r1_cot_data_test.json'

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, ensure_ascii=False, indent=4)

    logging.info(f"Results written to {output_file}")


if __name__ == "__main__":
    main()
