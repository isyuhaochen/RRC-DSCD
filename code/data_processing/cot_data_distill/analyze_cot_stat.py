import json
import logging
from collections import Counter
import statistics

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


def load_docs_evis_infos(path, pos_only=False):
    """
    Load positive and optionally negative reasoning content from a JSON file.

    Args:
        path (str): Path to the JSON file.
        pos_only (bool): Whether to load only positive examples.

    Returns:
        List[str]: A list of reasoning content strings.
    """
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    reason_content = []

    # Load positive examples
    positive_data = data.get("pos", {})
    logging.info(f"Number of positive examples: {len(positive_data)}")
    reason_content.extend(
        entry.get("reasoning_content", "") for entry in positive_data.values()
    )

    if not pos_only:
        # Load negative examples
        negative_data = data.get("neg", {})
        logging.info(f"Number of negative examples: {len(negative_data)}")
        reason_content.extend(
            entry.get("reasoning_content", "") for entry in negative_data.values()
        )

    return reason_content


def analyze_string_lengths(strings):
    """
    Analyze the length and word count statistics of a list of strings.

    Args:
        strings (List[str]): List of strings to analyze.

    Returns:
        Dict[str, float | int]: Summary statistics.
    """
    lengths = [len(s) for s in strings]
    word_counts = [len(s.split()) for s in strings]

    return {
        "average_length": statistics.mean(lengths),
        "median_length": statistics.median(lengths),
        "mode_length": statistics.mode(lengths),
        "min_length": min(lengths),
        "max_length": max(lengths),
        "average_words": statistics.mean(word_counts),
        "median_words": statistics.median(word_counts),
        "mode_words": statistics.mode(word_counts),
        "min_words": min(word_counts),
        "max_words": max(word_counts),
        # Optionally include full distributions:
        # "length_distribution": dict(Counter(lengths)),
        # "word_distribution": dict(Counter(word_counts))
    }


def main():
    input_path = "data/r1_cot_data_test.json"
    reasoning_content = load_docs_evis_infos(input_path, pos_only=False)

    stats = analyze_string_lengths(reasoning_content)
    logging.info("Text statistics:\n%s", json.dumps(stats, indent=4, ensure_ascii=False))


if __name__ == "__main__":
    main()
