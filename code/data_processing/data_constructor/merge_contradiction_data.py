import os
import json


def count_json_entries_in_folder(folder_path):
    """
    Count the total number of JSON entries in all .jsonl files within a folder.

    Args:
        folder_path (str): Path to the folder containing .jsonl files.

    Returns:
        int: Total number of JSON entries.
        list[dict]: List of all parsed JSON objects.
    """
    total_count = 0
    all_entries = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".jsonl"):
            file_path = os.path.join(folder_path, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    for line in file:
                        all_entries.append(json.loads(line.strip()))
                        total_count += 1
            except Exception as e:
                print(f"Failed to process file {file_path}: {e}")

    return total_count, all_entries


def process_json_entries(entries, output_path="data/pos.jsonl"):
    """
    Process a list of JSON entries and write valid samples to a new .jsonl file.

    Args:
        entries (list[dict]): A list of JSON objects to process.
        output_path (str): Path to output .jsonl file.
    """
    doc_id_tracker = {}
    valid_count = 0

    with open(output_path, "w", encoding="utf-8") as out_file:
        for entry in entries:
            try:
                doc_text = entry["doc"]
                doc_prefix = doc_text[:200]  # Use first 200 chars to construct ID
                doc_id_tracker[doc_prefix] = doc_id_tracker.get(doc_prefix, 0) + 1

                new_record = {
                    "contra_plug": entry["method"],
                    "contra_type": entry["contradiction_type"],
                    "contradiction_reason": entry["contradiction_reason"],
                    "text": doc_text,
                    "unique_id": f"{entry['doc_id'].split('_')[0]}_{abs(hash(doc_prefix))}_{doc_id_tracker[doc_prefix]}"
                }

                method = entry["method"]
                if method == "content_insert":
                    new_record.update({
                        "contra_list": [
                            entry["contradicted_sentence"],
                            entry["original_sentence"]
                        ] + entry.get("other_contradictory_sentences", []),
                        "original_sentence": entry["original_sentence"],
                        "contradicted_sentence": entry["contradicted_sentence"]
                    })
                elif method == "content_replace":
                    new_record.update({
                        "contra_list": [
                            entry["modified_sentence"]
                        ] + entry.get("other_contradictory_sentences", []),
                        "original_sentence": entry["original_sentence"],
                        "modified_sentence": entry["modified_sentence"]
                    })
                elif method == "content_swap":
                    new_record.update({
                        "contra_list": entry["modified_sentence_order"] + entry.get("other_contradictory_sentences", []),
                        "original_sentence_order": entry["original_sentence_order"],
                        "modified_sentence_order": entry["modified_sentence_order"]
                    })
                elif method == "content_delete":
                    new_record.update({
                        "contra_list": [
                            entry["sentencesA"],
                            entry["sentencesC"]
                        ] + entry.get("other_contradictory_sentences", []),
                        "sentencesA": entry["sentencesA"],
                        "sentencesB": entry["sentencesB"],
                        "sentencesC": entry["sentencesC"]
                    })
                else:
                    continue  # Unrecognized method

                if len(new_record["contra_list"]) > 1:
                    out_file.write(json.dumps(new_record, ensure_ascii=False) + "\n")
                    valid_count += 1
            except KeyError as e:
                print(f"Missing key in entry: {e}")
            except Exception as e:
                print(f"Error processing entry: {e}")

    print(f"Valid JSON entries written: {valid_count}")


if __name__ == "__main__":
    # Define your data folder path here
    input_folder = "data/contradiciton_data/"


    # Step 1: Read and count entries
    total_entries, parsed_entries = count_json_entries_in_folder(input_folder)
    print(f"Total JSON entries found: {total_entries}")

    # Step 2: Process and write output
    process_json_entries(parsed_entries)
