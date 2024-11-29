import os
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


def extract_input_feature(image_path, config, extractor_fn):
    """Extract global feature for the input image."""
    # Load and preprocess image
    pil_im = utils.RgbLoader(image_path)
    im = np.array(pil_im)

    # Extract features
    extracted_features = extractor_fn(im, resize_factor=1.0)
    return extracted_features['global_descriptor']


def retrieve_top_k(input_feature, index_features, image_paths, top_k):
    """Retrieve top_k similar images from index."""
    similarities = np.dot(index_features, input_feature)
    top_k_indices = np.argsort(-similarities)[:top_k]
    return [(image_paths[i], similarities[i]) for i in top_k_indices]


def main(image_path, index_features_dir, delf_config_path, top_k=5):
    """Main function to perform retrieval."""
    # Load DELF configuration
    config = delf_config_pb2.DelfConfig()
    with tf.io.gfile.GFile(delf_config_path, 'r') as f:
        text_format.Parse(f.read(), config)

    # Initialize extractor
    extractor_fn = extractor.MakeExtractor(config)

    # Extract input image feature
    input_feature = extract_input_feature(image_path, config, extractor_fn)

    # Load index features
    index_features, image_paths = load_index_features(index_features_dir)

    # Retrieve top_k results
    results = retrieve_top_k(input_feature, index_features, image_paths, top_k)

    return results


if __name__ == "__main__":
    # Example usage
    IMAGE_PATH = '/content/drive/MyDrive/5_NCKH/TCGA-data-DELG/TCGA-Images/TCGA-LGG_TCGA-DU-7007-01A-01-BS1.jp2_patch_3.png'
    INDEX_FEATURES_DIR = '/content/drive/MyDrive/5_NCKH/TCGA-data-DELG/TCGA_features/index'
    DELF_CONFIG_PATH = 'r50delg_gld_config.pbtxt'
    TOP_K = 5

    top_results = main(IMAGE_PATH, INDEX_FEATURES_DIR, DELF_CONFIG_PATH, TOP_K)

    print("Top K similar images:")
    for i, (path, score) in enumerate(top_results):
        print(f"{i+1}: {path} with similarity {score:.4f}")
