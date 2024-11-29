import json
from collections import defaultdict

def extract_disease_name(picture_name):
    """Extract disease name from a picture name."""
    return picture_name.split("_")[0]  # Assuming disease name is the first part


def calculate_metrics(results, k_values, u_values):
    """Calculate metrics for retrieval results."""
    metrics = {
        "recall@k": {k: [] for k in k_values},
        "precision@k": {k: [] for k in k_values},
        "f1@k": {k: [] for k in k_values},
        "map@u": {u: [] for u in u_values},
        "mrr@u": {u: [] for u in u_values},
    }

    for query_result in results:
        query_picture = query_result["picture"]
        query_disease = extract_disease_name(query_picture)
        retrieved_pictures = query_result["top200"]

        # Generate binary relevance list (1 if positive, 0 if negative)
        relevance = [
            1 if extract_disease_name(item["picture"]) == query_disease else 0
            for item in retrieved_pictures
        ]

        # Compute Recall@K, Precision@K, F1@K
        for k in k_values:
            top_k_relevance = relevance[:k]
            true_positives = sum(top_k_relevance)
            all_positives = relevance.count(1)

            recall = true_positives / min(all_positives, k) if min(all_positives, k) > 0 else 0
            precision = true_positives / k if k > 0 else 0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0
            )

            metrics["recall@k"][k].append(recall)
            metrics["precision@k"][k].append(precision)
            metrics["f1@k"][k].append(f1)

        # Compute MAP@U and MRR@U
        for u in u_values:
            top_u_relevance = relevance[:u]

            # MAP@U
            relevant_so_far = 0
            precision_at_i = []
            for i, rel in enumerate(top_u_relevance, 1):
                if rel == 1:
                    relevant_so_far += 1
                    precision_at_i.append(relevant_so_far / i)
            average_precision = sum(precision_at_i) / relevant_so_far if relevant_so_far > 0 else 0
            metrics["map@u"][u].append(average_precision)

            # MRR@U
            reciprocal_rank = 0
            for i, rel in enumerate(top_u_relevance, 1):
                if rel == 1:
                    reciprocal_rank = 1 / i
                    break
            metrics["mrr@u"][u].append(reciprocal_rank)

    # Compute mean for each metric
    for key in metrics:
        for subkey in metrics[key]:
            metrics[key][subkey] = sum(metrics[key][subkey]) / len(metrics[key][subkey])

    return metrics


if __name__ == "__main__":
    # Load results from JSON
    RESULTS_JSON_PATH = '/content/drive/MyDrive/5_NCKH/retrieval_results_pictures.json'
    with open(RESULTS_JSON_PATH, 'r') as f:
        retrieval_results = json.load(f)

    # Define evaluation parameters
    K_VALUES = [1, 3, 5, 10]
    U_VALUES = [5, 10]

    # Calculate metrics
    evaluation_metrics = calculate_metrics(retrieval_results, K_VALUES, U_VALUES)

    # Print results
    print("Evaluation Metrics:")
    for metric, values in evaluation_metrics.items():
        print(f"{metric}:")
        for k_or_u, value in values.items():
            print(f"  {k_or_u}: {value:.4f}")
