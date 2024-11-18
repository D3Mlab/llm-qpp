import pickle
import torch
import logging

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("EmbeddingComparison")

def compare_pickle_files(file1, file2):
    with open(file1, 'rb') as f1, open(file2, 'rb') as f2:
        index = 0
        try:
            while index < 5:
                # Load entries from both files
                data1 = pickle.load(f1)
                data2 = pickle.load(f2)

                # Print unpickled data
                logger.info(f"Data from file1 at index {index}: {data1}")
                logger.info(f"Data from file2 at index {index}: {data2}")

                # Compare doc IDs
                if data1['doc_id'] != data2['doc_id']:
                    logger.error(f"Mismatch at index {index}: doc_id {data1['doc_id']} != {data2['doc_id']}")
                    return False

                # Compare embeddings
                if not torch.allclose(data1['embedding'], data2['embedding'], atol=1e-6):
                    logger.error(f"Mismatch in embeddings at index {index} for doc_id {data1['doc_id']}")
                    logger.error(f"Embedding from file1: {data1['embedding']}")
                    logger.error(f"Embedding from file2: {data2['embedding']}")
                    return False

                index += 1

        except EOFError:
            logger.info("Reached end of both files.")

    logger.info("All entries match between the two files.")
    return True

if __name__ == "__main__":
    # Define file paths
    test_file_path = "data/toy_furniture/synonyms/TEST_collection_RandomEmbedder_.pkl"
    original_file_path = "data/toy_furniture/synonyms/collection_RandomEmbedder_.pkl"

    # Compare the files
    are_files_equal = compare_pickle_files(test_file_path, original_file_path)
    if are_files_equal:
        logger.info("Test passed: The files are identical.")
    else:
        logger.error("Test failed: The files have differences.")

