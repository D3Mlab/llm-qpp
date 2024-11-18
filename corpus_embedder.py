import json
import embedding
import pickle
from utils.setup_logging import setup_logging

def embed_corpus_jsonl(corpus_path, emb_path, embedder, logger, batch_size = 1):
#read "corpus_path" jsonl of format {"doc_id1" : XXX, 'text': XXX} one doc at a time
#compute doc embedding via an embedder.embed(text) which calls an instantiated embedder to return a torch tensor
#append docID and embedding to "emb_path" pickle file by pickling {"doc_id": doc_id, "embedding": tensor}

    processed_count = 0
    batch_texts = []
    batch_ids = []

    with open(corpus_path, 'r') as f, open(emb_path, 'wb') as emb_file:
        for line in f:
            doc = json.loads(line)
            doc_id, text = doc['docID'], doc['text']
            batch_ids.append(doc_id)
            batch_texts.append(text)

            # If batch is full, process it
            if len(batch_texts) == batch_size:
                embeddings = embedder.embed(batch_texts)
                for doc_id, embedding in zip(batch_ids, embeddings):
                    pickle.dump({"doc_id": doc_id, "embedding": embedding}, emb_file)
                batch_texts = []
                batch_ids = []
                processed_count += batch_size
                logger.info(f"Processed {processed_count} documents so far...")

        # Process any remaining documents
        if batch_texts:
            embeddings = embedder.embed(batch_texts)
            for doc_id, embedding in zip(batch_ids, embeddings):
                pickle.dump({"doc_id": doc_id, "embedding": embedding}, emb_file)
            processed_count += len(batch_texts)

        # Write total count at the end
        logger.info(f"Total documents embedded: {processed_count}")


if __name__ == "__main__":

    #model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    model_name = 'Alibaba-NLP/gte-large-en-v1.5'
    embedder = embedding.HuggingFaceEmbedder(model_name = model_name)
    model_name = model_name.replace('/', '-')

    #model_name = 'text-embedding-3-small' 
    #embedder = embedding.OpenAIEmbedder(model_name = model_name)

    #model_name = 'random' 
    #embedder = embedding.RandomEmbedder(model_name = model_name)

    data_path = "data/MS-MARCO/subset_q10_d100/"

    #setup logging
    config = {
        "logging": {
            "level": "INFO",
            "log_file": f"{data_path}embedding_log_{model_name}.txt",
        }
    }

    logger = setup_logging(f"{model_name}", config)

    corpus_path = f"{data_path}collection.jsonl"
    emb_path = f"{data_path}collection_{model_name}.pkl"

    embed_corpus_jsonl(corpus_path, emb_path, embedder, logger, batch_size = 1)

    #test:
    # Read the embeddings back from the pickle file
    
    #with open(emb_path, 'rb') as emb_file:
    #    try:
    #        while True:
    #            data = pickle.load(emb_file)
    #            logger.info(
    #               f"Read embedding for doc_id: {data['doc_id']}, \
    #               embedding: {data['embedding']}, \
    #               length: {data['embedding'].shape}")
    #    except EOFError:
    #       pass
    #logger.info("Finished reading all embeddings from the pickle file.")