import json
import embedders
import pickle
from utils.setup_logging import setup_logging

def embed_corpus_jsonl(corpus_path, emb_path, embedder, logger):
#read "corpus_path" jsonl of format {"doc_id1" : XXX, 'text': XXX} one doc at a time
#compute doc embedding via an embedder.embed(text) which calls an instantiated embedder to return a torch tensor
#append docID and embedding to "emb_path" pickle file by pickling {"doc_id": doc_id, "embedding": tensor}

    processed_count = 0

    with open(corpus_path, 'r') as f, open(emb_path, 'wb') as emb_file:
        for line in f:
            doc = json.loads(line)
            doc_id, text = doc['docID'], doc['text']
            embedding = embedder.embed(text)
            pickle.dump({"doc_id": doc_id, "embedding": embedding}, emb_file)

            processed_count += 1
            if processed_count % 100 == 0:
                logger.info(f"Processed {processed_count} documents so far...")

        # Write total count at the end
        logger.info(f"Total documents embedded: {processed_count}")


if __name__ == "__main__":

    embedder = embedders.RandomEmbedder()

    data_path = "data/toy_furniture/synonyms/"

    #setup logging
    config = {
        "logging": {
            "level": "INFO",
            "log_file": f"{data_path}embedding_log_{embedder.__class__.__name__}.txt",
        }
    }

    logger = setup_logging(f"{embedder.__class__.__name__}", config)

    corpus_path = f"{data_path}collection.jsonl"
    emb_path = f"{data_path}collection_{embedder.__class__.__name__}.pkl"

    embed_corpus_jsonl(corpus_path, emb_path, embedder, logger)
