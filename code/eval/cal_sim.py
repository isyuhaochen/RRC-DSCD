import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def calculate_cosine_similarity(arr1, arr2):
    """
    Compute the cosine similarity between two arrays.
    
    Args:
        arr1 (array-like): First array.
        arr2 (array-like): Second array.
    
    Returns:
        float: Cosine similarity between arr1 and arr2.
    """
    arr1 = np.array(arr1).reshape(1, -1)
    arr2 = np.array(arr2).reshape(1, -1)
    similarity = cosine_similarity(arr1, arr2)
    return similarity[0][0]


def read_data(file_path):
    """
    Read predictions from a .npz file.
    
    Args:
        file_path (str): Path to the .npz file.
    
    Returns:
        tuple: Non-verified and verified predictions.
    """
    data = np.load(file_path)
    non_verified_predictions = data['non_verified']
    verified_predictions = data['verified']
    return non_verified_predictions, verified_predictions


def count_matching_elements_ratio(list1, list2):
    """
    Calculate the ratio of matching elements between two lists.
    
    Args:
        list1 (list): First list.
        list2 (list): Second list.
    
    Returns:
        float: Ratio of matching elements.
    """
    return sum(1 for a, b in zip(list1, list2) if a == b) / len(list1)


def process_predictions(path_list):
    """
    Process multiple prediction files to calculate cosine similarity and matching ratio.
    
    Args:
        path_list (list of str): List of file paths to .npz files.
    
    Returns:
        tuple: Lists of cosine similarities and matching ratios for non-verified and verified predictions.
    """
    non_verified_predictions_all = []
    verified_predictions_all = []
    non_verified_similarity_all = []
    verified_similarity_all = []
    non_verified_matching_count_all = []
    verified_matching_count_all = []

    for file_path in path_list:
        non_verified_predictions, verified_predictions = read_data(file_path)
        non_verified_predictions_all.append(non_verified_predictions)
        verified_predictions_all.append(verified_predictions)

    for i in range(len(path_list)):
        for j in range(i + 1, len(path_list)):  # Ensure j > i to avoid duplicate combinations
            non_verified_similarity = calculate_cosine_similarity(non_verified_predictions_all[i], non_verified_predictions_all[j])
            verified_similarity = calculate_cosine_similarity(verified_predictions_all[i], verified_predictions_all[j])

            non_verified_similarity_all.append(non_verified_similarity)
            verified_similarity_all.append(verified_similarity)

            # Calculate matching element ratios
            non_verified_matching_count = count_matching_elements_ratio(non_verified_predictions_all[i], non_verified_predictions_all[j])
            verified_matching_count = count_matching_elements_ratio(verified_predictions_all[i], verified_predictions_all[j])

            non_verified_matching_count_all.append(non_verified_matching_count)
            verified_matching_count_all.append(verified_matching_count)

    return (non_verified_similarity_all, verified_similarity_all,
            non_verified_matching_count_all, verified_matching_count_all)


def print_statistics(similarity_all, matching_count_all, label):
    """
    Print the statistics of cosine similarity and matching element ratios.
    
    Args:
        similarity_all (list): List of cosine similarity values.
        matching_count_all (list): List of matching element ratios.
        label (str): Label to identify the prediction type ('non-verified' or 'verified').
    """
    average_similarity = np.mean(similarity_all)
    std_similarity = np.std(similarity_all)

    average_matching_count = np.mean(matching_count_all)
    std_matching_count = np.std(matching_count_all)

    print(f"Average cosine similarity for {label} predictions: {average_similarity}")
    print(f"Standard deviation of cosine similarity for {label} predictions: {std_similarity}")
    print(f"Average matching elements ratio for {label} predictions: {average_matching_count}")
    print(f"Standard deviation of matching elements ratio for {label} predictions: {std_matching_count}")


def main():
    # List of file paths
    path_list = [
        'npz/llama_sft_plus_20250501_113206.npz',
        'npz/llama_sft_plus_20250501_113242.npz',
        'llama_sft_plus_20250501_113349.npz'
    ]

    # Process the predictions
    (non_verified_similarity_all, verified_similarity_all,
     non_verified_matching_count_all, verified_matching_count_all) = process_predictions(path_list)

    # Print statistics for non-verified predictions
    print_statistics(non_verified_similarity_all, non_verified_matching_count_all, 'non-verified')

    # Print statistics for verified predictions
    print_statistics(verified_similarity_all, verified_matching_count_all, 'verified')


if __name__ == "__main__":
    main()
