import json
import pandas as pd
import random
import os

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


def process_data(data_list, dataset_name, id_dict, file_obj):
    """
    Process a list of documents and write them as JSONL entries.

    Args:
        data_list (list): List of textual documents.
        dataset_name (str): Name of the dataset, used for generating unique IDs.
        id_dict (dict): Dictionary to track document prefixes and their counts.
        file_obj (file): Opened file object for writing JSONL entries.
    """
    for doc in data_list:
        doc_prefix = doc[:200]  # Use the first 200 characters as a unique key
        id_dict[doc_prefix] = id_dict.get(doc_prefix, 0) + 1
        unique_id = f"{dataset_name}_{abs(hash(doc_prefix))}_{id_dict[doc_prefix]}"

        json_entry = {
            'unique_id': unique_id,
            'text': doc
        }
        file_obj.write(json.dumps(json_entry, ensure_ascii=False) + "\n")


def main():
    id_tracker = {}


    with open("test_data/data_neg.jsonl", "w", encoding="utf-8") as f:
        # Process the StorySumm test set
        storysumm_data = read_data('storysumm', data_path='data/raw_data/', read_type='test')
        print(f"Number of StorySumm entries: {len(storysumm_data)}")
        process_data(storysumm_data, 'storysumm', id_tracker, f)

        # Process the RepliQA test set (via read_data)
        repliqa_data = read_data('repliqa', data_path='data/raw_data/', read_type='test')
        print(f"Number of RepliQA (via read_data) entries: {len(repliqa_data)}")
        process_data(repliqa_data, 'repliqa', id_tracker, f)



if __name__ == "__main__":
    main()
