import json
import random
from collections import defaultdict

# Set random seed for reproducibility
random.seed(42)


def extract_id(unique_id):
    """
    Extract the base ID from a unique_id string, assuming the format 'X_Y_Z...'
    Returns 'X_Y' as the identifier.
    """
    parts = unique_id.split('_')
    return f"{parts[0]}_{parts[1]}"


def load_jsonl(file_path, count_dict):
    """
    Load a JSONL file into a dictionary and update the count_dict with ID frequencies.
    """
    data = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            json_obj = json.loads(line.strip())
            unique_id = json_obj['unique_id']
            id_key = extract_id(unique_id)
            count_dict[id_key] += 1
            data[unique_id] = json_obj
    return data


def split_data(data, id_subset):
    """
    Split data into training and testing sets based on whether their IDs are in the provided subset.
    """
    train_data, test_data = {}, {}
    for key, value in data.items():
        if extract_id(key) in id_subset:
            test_data[key] = value
        else:
            train_data[key] = value
    return train_data, test_data


def save_json(data_dict, file_path):
    """
    Save a dictionary to a JSON file with pretty formatting.
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data_dict, f, ensure_ascii=False, indent=4)


def main():
    # Initialize counters for positive and negative datasets
    pos_data_count = defaultdict(int)
    neg_data_count = defaultdict(int)

    # Load datasets
    pos_data = load_jsonl('test_data/data_pos.jsonl', pos_data_count)
    neg_data = load_jsonl('test_data/data_neg.jsonl', neg_data_count)

    # Select IDs that appear only once
    pos_ids = [k for k, v in pos_data_count.items() if v == 1]
    neg_ids = [k for k, v in neg_data_count.items() if v == 1]

    # Randomly sample up to 1000 IDs from each
    pos_sampled_ids = random.sample(pos_ids, min(1000, len(pos_ids)))
    neg_sampled_ids = random.sample(neg_ids, min(1000, len(neg_ids)))

    # Split into train and test sets
    pos_train, pos_test = split_data(pos_data, pos_sampled_ids)
    neg_train, neg_test = split_data(neg_data, neg_sampled_ids)

    # Merge datasets
    merged_sft = {'pos': pos_train, 'neg': neg_train}
    merged_rl = {'pos': pos_test, 'neg': neg_test}

    # Save output
    save_json(merged_sft, 'data/merged_sft.json')
    save_json(merged_rl, 'data/merged_rl.json')

    print("Data processing complete." )


if __name__ == "__main__":
    main()
