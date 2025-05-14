import json
import logging

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

def load_json(file_path):
    """Load JSON data from a file and return it"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        logging.info(f"Successfully loaded data from {file_path}")
        return data
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        return {}
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON from {file_path}")
        return {}

def save_json(data, file_path):
    """Save data to a JSON file"""
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)
        logging.info(f"Data successfully saved to {file_path}")
    except Exception as e:
        logging.error(f"Error saving data to {file_path}: {e}")

def merge_datasets(data1, data2):
    """Merge two datasets"""
    return {
        'pos': {**data1.get('pos', {}), **data2.get('pos', {})},
        'neg': {**data1.get('neg', {}), **data2.get('neg', {})}
    }

def main():
    # Load dataset 1
    data1 = load_json('r1_cot_data_rl.json')

    # Load dataset 2
    data2 = load_json('r1_cot_data_sft.json')

    # Merge datasets
    merged_data = merge_datasets(data1, data2)

    # Save the merged dataset
    save_json(merged_data, 'merged_data.json')

if __name__ == "__main__":
    main()
