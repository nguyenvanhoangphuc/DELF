import os
import json
from collections import defaultdict

def extract_picture_name(feature_path):
    """Extract picture name from a feature file path."""
    base_name = os.path.basename(feature_path)
    # Remove ".delg_global" and "patch_X"
    picture_name = "_".join(base_name.split("_")[:-2])
    return picture_name


def rank_by_pictures(results, top_k=200):
    """Re-rank results by pictures instead of patches."""
    ranked_results = []

    for query_result in results:
        query_feature_path = query_result["query"]
        query_picture = extract_picture_name(query_feature_path)

        # Group patches by picture for top200 results
        picture_scores = defaultdict(list)
        for item in query_result["top200"]:
            picture_name = extract_picture_name(item["path"])
            picture_scores[picture_name].append(item["score"])

        # Aggregate scores for each picture (max score method)
        picture_aggregated = [
            {"picture": picture, "score": max(scores)}
            for picture, scores in picture_scores.items()
        ]

        # Sort pictures by score
        picture_aggregated.sort(key=lambda x: -x["score"])

        # Take top_k pictures
        top_pictures = picture_aggregated[:top_k]

        # Store results for this query
        ranked_results.append({
            "picture": query_picture,
            "top200": top_pictures
        })

    return ranked_results


if __name__ == "__main__":
    # Load results from JSON
    INPUT_JSON_PATH = '/content/drive/MyDrive/5_NCKH/retrieval_results.json'
    OUTPUT_JSON_PATH = '/content/drive/MyDrive/5_NCKH/retrieval_results_pictures.json'

    with open(INPUT_JSON_PATH, 'r') as f:
        retrieval_results = json.load(f)

    # Re-rank results by pictures
    ranked_results = rank_by_pictures(retrieval_results)

    # Save re-ranked results to JSON
    with open(OUTPUT_JSON_PATH, 'w') as f:
        json.dump(ranked_results, f, indent=2)

    print(f"Re-ranked results saved to {OUTPUT_JSON_PATH}")
