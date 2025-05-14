import os
import sys
import re
import ast
import json
import argparse
import torch
import string
import random
import logging
import pandas as pd
  
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "../../"))

from prompt import get_statement,content_insert,content_replace,content_swap,content_delete, test_example
from model import get_model_response
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm 

def read_data(data_name: str, data_path: str = 'data/raw_data/', read_type: str = 'train') -> list:
    """
    Reads data from a specified source and returns a subset based on the requested read type.

    Args:
        data_name (str): The name of the data source ('test_example', 'storysumm', or 'repliqa').
        data_path (str): The directory path where the data is located. Default is 'raw_data/'.
        read_type (str): Specifies whether to return the 'train' or 'test' portion of the data.
                         'train' returns the first 80%, 'test' returns the remaining 20%.

    Returns:
        list: A list of unique data entries.
    """
    random.seed(42)  # For reproducibility
    unique_data_list = []

    # Load the appropriate data based on the data_name
    if data_name == "test_example":
        return [test_example]  # Assuming test_example is predefined elsewhere
    elif data_name == "storysumm":
        file_path = os.path.join(data_path, f"{data_name}.json")
        with open(file_path, "r") as file:
            data = json.load(file)
        # Extract stories from data
        unique_data_list = list({value['story'] for value in data.values()})
    elif data_name == "repliqa":
        file_path = os.path.join(data_path, "repliqa_2-00000-of-00001.parquet")
        df = pd.read_parquet(file_path)
        unique_data_list = df['document_extracted'].unique().tolist()

    # Sort the data and determine the 80% cutoff for training data
    sorted_list = sorted(unique_data_list)
    top_80_percent = int(len(sorted_list) * 0.8)

    # Return the requested portion of the data
    return sorted_list[:top_80_percent] if read_type == 'train' else sorted_list[top_80_percent:]

def extract_list_from_string(s: str) -> list:
    """
    Extracts a list from a string representation of a list.

    Args:
        s (str): The input string that may contain a list.

    Returns:
        list: The extracted list if found and valid, otherwise None.
    """
    # Match the first occurrence of a list-like structure
    match = re.search(r'\[.*?\]', s)
    
    if match:
        list_str = match.group(0)
        try:
            # Safely evaluate the matched string as a Python literal
            result = ast.literal_eval(list_str)
            # Return the result only if it's a valid list
            return result if isinstance(result, list) else None
        except (SyntaxError, ValueError):
            # Return None in case of invalid literal
            return None
    
    return None

def judge(original_text: str, new_text: str, ppl_model, ppl_tokenizer, device) -> bool:
    """
    Evaluates whether the new text has similar perplexity to the original text.

    Args:
        original_text (str): The original text.
        new_text (str): The modified text.
        ppl_model: The model used to calculate perplexity.
        ppl_tokenizer: The tokenizer used for text preprocessing.
        device: The device on which the model is running (CPU or GPU).

    Returns:
        bool: True if the new text has a perplexity ratio of less than 1.01 compared to the original text, 
              False otherwise or if an error occurs during perplexity calculation.
    """
    # Calculate perplexity for both original and new text
    original_ppl = calculate_perplexity(original_text, ppl_model, ppl_tokenizer, device)
    new_ppl = calculate_perplexity(new_text, ppl_model, ppl_tokenizer, device)
    
    # If either perplexity is NaN, return False
    if original_ppl != original_ppl or new_ppl != new_ppl:  # Check for NaN using != NaN
        return False
    
    # Calculate perplexity ratio and determine if it's within an acceptable range
    perplexity_ratio = new_ppl / original_ppl
    if perplexity_ratio < 1.01:
        return True

    return False

def calculate_perplexity(text: str, ppl_model, ppl_tokenizer, device) -> float:
    """
    Computes the perplexity of a given text using a specified model and tokenizer.

    Args:
        text (str): The input text to evaluate.
        ppl_model: The pre-trained model used to calculate perplexity.
        ppl_tokenizer: The tokenizer to preprocess the input text.
        device: The device on which the model is running.

    Returns:
        float: The calculated perplexity or NaN if an error occurs.
    """
    try:
        # Tokenize the input text and convert it to a PyTorch tensor
        input_ids = ppl_tokenizer.encode(
            text, add_special_tokens=True, max_length=8192, truncation=True, return_tensors="pt"
        ).to(device)
        
        # Disable gradient computation for efficiency
        with torch.no_grad():
            outputs = ppl_model(input_ids, labels=input_ids)
            loss = outputs.loss
        
        # Compute perplexity from the loss value
        perplexity = torch.exp(loss).item()

        # Return perplexity if it's a valid value, otherwise return NaN
        return perplexity if perplexity < float('inf') else float('nan')

    except Exception as e:
        # Log the error and return NaN in case of an exception
        print(f"Error in perplexity calculation: {e}")
        return float('nan')
 
def judge_sentence(example, article):
    """
    Checks if a given example (string or list of strings) is present in the provided article.

    Args:
        example (str or list): A string or list of strings to search for in the article.
        article (str): The article in which to search for the example.

    Returns:
        str or list: The matching string(s) if found in the article; 
                     None if the example is not found or is of an unsupported type.
    """
    if isinstance(example, str):
        # If the example is a string, check if it exists in the article
        return example if example in article else None

    if isinstance(example, list):
        # If the example is a list, return the items that exist in the article
        return [item for item in example if item in article]

    # Return None if the example is neither a string nor a list
    return None

def parse_content(model_ans, doc, method):
    """
    Parse the model's answer to extract relevant content based on the provided method.
    
    Args:
        model_ans (str): The model's answer containing the content to be parsed.
        doc (str): The document to check for sentence validity.
        method (str): The method for content manipulation. Options are:
            'content_insert', 'content_replace', 'content_swap', 'content_delete'.
    
    Returns:
        dict or None: A dictionary of parsed content if valid, else None.
    """
    # Match the content inside curly braces
    match = re.search(r"\{(.*?)\}", model_ans, re.DOTALL)
    
    # Define required keys based on the specified method
    if method == "content_insert":
        required_keys = [
            'original_sentence', 'contradicted_sentence', 'insert_position_sentence', 
            'next_sentence_after_insert', 'other_contradictory_sentences', 
            'contradiction_type', 'contradiction_reason'
        ]
        required_keys_in_doc = [required_keys[0], required_keys[2], required_keys[3]]
        sentences_list = [required_keys[4]]
    
    elif method == "content_replace":
        required_keys = [
            'original_sentence', 'modified_sentence', 'other_contradictory_sentences', 
            'contradiction_type', 'contradiction_reason'
        ]
        required_keys_in_doc = [required_keys[0]]
        sentences_list = [required_keys[2]]
    
    elif method == "content_swap":
        required_keys = [
            'original_sentence_order', 'modified_sentence_order', 'other_contradictory_sentences', 
            'contradiction_type', 'contradiction_reason'
        ]
        required_keys_in_doc = []
        sentences_list = [required_keys[0], required_keys[2]]
    
    elif method == "content_delete":
        required_keys = [
            'sentencesA', 'sentencesB', 'sentencesC', 'other_contradictory_sentences', 
            'contradiction_type', 'contradiction_reason'
        ]
        required_keys_in_doc = [required_keys[0], required_keys[1], required_keys[2]]
        sentences_list = [required_keys[3]]

    # If a match is found in the model's answer
    if match:
        json_text = match.group(0)
        try:
            json_item = json.loads(json_text)

            # Ensure the parsed JSON is not empty
            if not json_item:
                return None

            # Validate that all required keys exist and are non-empty
            for key in required_keys:
                if key not in json_item or (key != 'other_contradictory_sentences' and not json_item[key]):
                    return None

            # Check that the required keys exist in the document
            for key in required_keys_in_doc:
                if json_item[key] not in doc:
                    return None

            # Filter the sentences in `sentences_list` to ensure they exist in the document
            for sentence_list in sentences_list:
                json_item[sentence_list] = [
                    item for item in json_item[sentence_list] if item in doc
                ]

            # Apply additional validation rules based on the method
            if method == "content_replace":
                if json_item['modified_sentence'] == '' or not json_item[sentences_list[0]]:
                    return None
            elif method == "content_swap":
                if len(json_item['modified_sentence_order']) != len(json_item['original_sentence_order']):
                    return None
            elif method == "content_delete":
                if json_item['sentencesA'] == '' or json_item['sentencesB'] == '' or json_item['sentencesC'] == '':
                    return None
            elif method == "content_replace":
                if json_item['original_sentence'] in json_item['other_contradictory_sentences'] or json_item['modified_sentence'] in json_item['other_contradictory_sentences']:
                    return None

            return json_item
        except Exception:
            return None
    else:
        return None

def generate_random_string(length):
    """
    Generates a random string of a specified length composed of ASCII letters, digits, and punctuation characters.

    Args:
        length (int): The desired length of the random string.

    Returns:
        str: A random string of the specified length.
    """
    # Define the character set to include ASCII letters, digits, and punctuation
    charset = string.ascii_letters + string.digits + string.punctuation
    
    # Generate and return the random string by selecting random characters from the charset
    return ''.join(random.choice(charset) for _ in range(length))

def parse_args():
    """
    Parses command-line arguments for the GPT fake document generator.

    Returns:
        Namespace: A namespace object containing the parsed arguments.
    """
    # Initialize argument parser with a brief description
    parser = argparse.ArgumentParser(description='Self-Contradiction Document Generator')

    # Define arguments with corresponding types, default values, and descriptions
    parser.add_argument('--source_doc', type=str, default='test_example', choices=['test_example', 'storysumm', 'repliqa'],
                        help='Specify the type of document source to use (e.g., test_example, storysumm, or repliqa).')

    parser.add_argument('--extract_method', type=str, default='content_swap', choices=['content_insert', 'content_replace', 'content_swap', 'content_delete'],
                        help='Choose the document modification technique to apply (e.g., content_insert, content_replace, content_swap, or content_delete).')

    parser.add_argument('--contra_doc_path', type=str, default='output/contradiction_example.jsonl',
                        help='Path to the output file where the generated contradictory examples will be saved.')

    parser.add_argument('--CUDA_VISIBLE_DEVICES', type=str, default='1',
                        help='Specify which GPU device to use for processing.')

    parser.add_argument('--data_start', type=int, default=0,
                        help='Specify the starting index of the data subset to process.')

    parser.add_argument('--data_end', type=int, default=100000000,
                        help='Specify the ending index of the data subset to process.')

    parser.add_argument('--model_path', type=str, default='/data2/LLM_Model/Meta-Llama-3.1-8B-Instruct',
                        help='Path to the pre-trained model directory.')

    # Parse and return the arguments
    return parser.parse_args()


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Parse command-line arguments
    args = parse_args()
    
    # Load the data based on the source document type
    data_name = args.source_doc
    data = read_data(data_name, read_type='train')
    data = data[args.data_start:args.data_end]

    # Log data loading information
    logging.info(f"Loaded data: {data_name}, count = {len(data)}")
    logging.info(f"Data range: start = {args.data_start}, end = {args.data_end}")
    
    # Set up CUDA device for model execution
    os.environ["CUDA_VISIBLE_DEVICES"] = args.CUDA_VISIBLE_DEVICES
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Load the pre-trained model and tokenizer
    ppl_tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    ppl_model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map="auto")
    logging.info('Loaded perplexity model')
    
     # Initialize counters
    total_count = 0
    access_count = 0

    # Process the documents
    for doc in tqdm(data, desc="Processing"):
        if args.extract_method in ["content_insert", "content_replace"]:
            text = get_statement() + doc
            model_ans = get_model_response(text)
            extract_list = extract_list_from_string(model_ans)
            extract_list = judge_sentence(extract_list, doc)
            if extract_list:
                doc_item = {"doc": doc, "Statement": extract_list}
                
                if args.extract_method == "content_insert":
                    for statement in doc_item["Statement"]:
                        prompt = content_insert.format(Article=doc, Statement=statement)
                        model_ans = get_model_response(prompt)
                        json_item = parse_content(model_ans, doc, args.extract_method)
                        
                        if json_item:
                            insert_pos = doc.find(json_item["insert_position_sentence"]) + len(json_item["insert_position_sentence"])
                            next_sentence_pos = doc.find(json_item["next_sentence_after_insert"])
                            
                            if next_sentence_pos - insert_pos == 1:
                                new_doc = doc[:insert_pos] + " " + json_item["contradicted_sentence"] + " " + doc[insert_pos:]
                                json_item['doc'] = new_doc
                                json_item['method'] = 'content_insert'
                                json_item['doc_id'] = f"{data_name}_"
                                total_count += 1

                                # Check with perplexity model
                                if judge(doc, new_doc, ppl_model, ppl_tokenizer, device):
                                    with open(args.contra_doc_path, "a", encoding="utf-8") as f:
                                        f.write(json.dumps(json_item, ensure_ascii=False) + "\n")
                                    access_count += 1

                elif args.extract_method == "content_replace":
                    for statement in doc_item["Statement"]:
                        prompt = content_replace.format(Article=doc, Statement=statement)
                        model_ans = get_model_response(prompt)
                        json_item = parse_content(model_ans, doc, args.extract_method)
                        
                        if json_item:
                            new_doc = doc.replace(json_item["original_sentence"], json_item["modified_sentence"], 1)
                            json_item['doc'] = new_doc
                            json_item['method'] = 'content_replace'
                            json_item['doc_id'] = f"{data_name}_"
                            total_count += 1

                            # Check with perplexity model
                            if judge(doc, new_doc, ppl_model, ppl_tokenizer, device):
                                with open(args.contra_doc_path, "a", encoding="utf-8") as f:
                                    f.write(json.dumps(json_item, ensure_ascii=False) + "\n")
                                access_count += 1

        elif args.extract_method == "content_swap":
            prompt = content_swap.format(Article=doc)
            model_ans = get_model_response(prompt)
            json_item = parse_content(model_ans, doc, args.extract_method)
            
            if json_item:
                list_len = len(json_item["original_sentence_order"])
                temp_list = [generate_random_string(10) for _ in range(list_len)]
                new_doc = doc
                
                # Perform sentence swapping
                for i in range(list_len):
                    new_doc = new_doc.replace(json_item["original_sentence_order"][i], temp_list[i])
                for i in range(list_len):
                    new_doc = new_doc.replace(temp_list[i], json_item["modified_sentence_order"][i])

                json_item['doc'] = new_doc
                json_item['method'] = 'content_swap'
                json_item['doc_id'] = f"{data_name}_"
                total_count += 1
                
                # Check with perplexity model
                if judge(doc, new_doc, ppl_model, ppl_tokenizer, device):
                    with open(args.contra_doc_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(json_item, ensure_ascii=False) + "\n")
                    access_count += 1
        
        elif args.extract_method == "content_delete":
            prompt = content_delete.format(Article=doc)
            model_ans = get_model_response(prompt)
            json_item = parse_content(model_ans, doc, args.extract_method)
            
            if json_item:
                new_doc = doc.replace(json_item["sentencesB"], "")
                json_item['doc'] = new_doc
                json_item['method'] = 'content_delete'
                json_item['doc_id'] = f"{data_name}_"
                total_count += 1
                
                # Check with perplexity model
                if judge(doc, new_doc, ppl_model, ppl_tokenizer, device):
                    with open(args.contra_doc_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(json_item, ensure_ascii=False) + "\n")
                    access_count += 1

    # Log final statistics
    logging.info(f"Total documents processed: {total_count}")
    logging.info(f"Documents with valid contradictions: {access_count}")
    logging.info("Processing complete.")