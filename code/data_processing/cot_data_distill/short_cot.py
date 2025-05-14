import json
import os
import sys
from tqdm import tqdm
from model import get_model_response

# Set up paths and add to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "../"))

# Define constants
MAX_REASON_WORD_COUNT = 500
PROMPT = 'Revise this reasoning process by removing irrelevant words, without losing information, and without altering the original descriptive logic and word order (Control within {max_word_num} words), please do not output anything else. text:{text} \n new_text:'

# Load the data
def load_data(file_path):
    """Load JSON data from a file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# Save the updated data to a file
def save_data(file_path, data):
    """Save data to a JSON file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)
    print(f"Data saved to {file_path}")

# Process and update reasoning content
def process_reasoning_content(data, max_word_count):
    """Process and update reasoning content based on the word count limit."""
    to_delete = {'pos': [], 'neg': []}
    
    for item_type in ['pos', 'neg']:
        for key, value in tqdm(data[item_type].items(), desc=f"Processing {item_type} items"):
            reasoning_content = value.get('reasoning_content', '')
            word_count = len(reasoning_content.split())

            if word_count > max_word_count:
                success = False
                for _ in range(10):
                    new_text = get_model_response(PROMPT.format(max_word_num=max_word_count, text=reasoning_content))
                    if len(new_text.split()) <= max_word_count:
                        success = True
                        break

                if success:
                    value['reasoning_content'] = new_text
                    print(f"Updated content for '{key}':\n{new_text}")
                else:
                    print(f"Deleting item '{key}' due to exceeding word limit.")
                    to_delete[item_type].append(key)

    # Delete items that exceeded the word limit
    for item_type in ['pos', 'neg']:
        for key in to_delete[item_type]:
            data[item_type].pop(key, None)
    
    return data

def main():
    # Load data
    data = load_data('r1_cot_data_test.json')

    # Process reasoning content
    updated_data = process_reasoning_content(data, MAX_REASON_WORD_COUNT)

    # Save the updated data
    save_data('r1_cot_test_new.json', updated_data)

    print("Processing complete.")

if __name__ == "__main__":
    main()
