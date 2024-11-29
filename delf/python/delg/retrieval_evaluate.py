import os
import json
import numpy as np
import tensorflow as tf
from delf import datum_io, extractor, utils
from google.protobuf import text_format
from delf import delf_config_pb2

# Constants
_DELG_GLOBAL_EXTENSION = '.delg_global'
_IMAGE_EXTENSION = '.png'


def load_index_features(index_features_dir):
    """Load global features from index directory."""
    feature_files = [f for f in os.listdir(index_features_dir) if f.endswith(_DELG_GLOBAL_EXTENSION)]
    index_features = []
    image_paths = []
    for feature_file in feature_files:
        feature_path = os.path.join(index_features_dir, feature_file)
        index_features.append(datum_io.ReadFromFile(feature_path))
        image_paths.append(feature_path)
    return np.array(index_features), image_paths


def extract_input_feature(feature_path):
    """Load global feature for a query image."""
    return datum_io.ReadFromFile(feature_path)


def retrieve_top_k(input_feature, index_features, image_paths, top_k):
    """Retrieve top_k similar images from index."""
    similarities = np.dot(index_features, input_feature)
    top_k_indices = np.argsort(-similarities)[:top_k]
    return [{"path": image_paths[i], "score": float(similarities[i])} for i in top_k_indices]


def retrieve_for_all_queries(query_features_dir, index_features_dir, delf_config_path, top_k=200):
    """Perform retrieval for all queries and save results to JSON."""
    # Load DELF configuration (not required for pre-extracted features)
    index_features, image_paths = load_index_features(index_features_dir)

    # Iterate over query features
    feature_files = [f for f in os.listdir(query_features_dir) if f.endswith(_DELG_GLOBAL_EXTENSION)]
    results = []
    for feature_file in feature_files:
        query_feature_path = os.path.join(query_features_dir, feature_file)
        input_feature = extract_input_feature(query_feature_path)
        top_k_results = retrieve_top_k(input_feature, index_features, image_paths, top_k)

        # Store results
        results.append({
            "query": query_feature_path,
            "top200": top_k_results
        })

    return results


def save_results_to_json(results, output_path):
    """Save retrieval results to a JSON file."""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    # Example usage
    QUERY_FEATURES_DIR = '/content/drive/MyDrive/5_NCKH/TCGA-data-DELG/TCGA_features/query'
    INDEX_FEATURES_DIR = '/content/drive/MyDrive/5_NCKH/TCGA-data-DELG/TCGA_features/index'
    DELF_CONFIG_PATH = 'r50delg_gld_config.pbtxt'  # Not used here
    OUTPUT_JSON_PATH = '/content/drive/MyDrive/5_NCKH/retrieval_results.json'
    TOP_K = 200

    # Perform retrieval
    retrieval_results = retrieve_for_all_queries(
        QUERY_FEATURES_DIR, INDEX_FEATURES_DIR, DELF_CONFIG_PATH, TOP_K
    )

    # Save results to JSON
    save_results_to_json(retrieval_results, OUTPUT_JSON_PATH)

    print(f"Retrieval results saved to {OUTPUT_JSON_PATH}")
