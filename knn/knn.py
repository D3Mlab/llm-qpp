from abc import ABC, abstractmethod
import pickle
import json
import torch
import heapq

class KNN(ABC):
    def __init__(self, config, corpus_emb_path):
        self.config = config
        self.logger = setup_logging(self.__class__.__name__, self.config)
        self.corpus_emb_path = corpus_emb_path

    @abstractmethod
    def get_top_k(self, query_embedding, similarity_fun, k, implementation = None):
         raise NotImplementedError("This method must be implemented by a subclass.")


class ExactKNN(KNN):

    def __init__(self, config, corpus_emb_path):
        super().__init__(config, corpus_emb_path)
        self.IMPLEMENTATION_METHODS = {
            'load_all': self.load_all,
            'load_individual': self.load_individual,
        }

    def get_top_k(self, query_embedding, similarity_fun_name, k, implementation):
        implementation_method = self.IMPLEMENTATION_METHODS.get(implementation,'load_all')
        return implementation_method(query_embedding, similarity_fun_name, k)


    def load_all(self, query_embedding, similarity_fun_name, top_k):
        """
        Loads all embeddings from the corpus, computes similarity with the query embedding,
        and returns the top_k most similar document IDs and the corresponding similarity scores.
        """
        doc_ids = []
        embeddings = []

        # Load all embeddings
        with open(self.corpus_emb_path, 'rb') as corpus_file:
            try:
                while True:
                    doc = pickle.load(corpus_file)
                    doc_ids.append(doc['doc_id'])
                    embeddings.append(doc['embedding'])
            except EOFError:
                self.logger.error("could not load all embeddings into memory")

        # Stack all embeddings into a tensor
        #todo - embeddings to device
        embeddings = torch.stack(embeddings)
        query_embedding = query_embedding.unsqueeze(0)

        # Compute similarity based on the selected function
        if similarity_fun_name == 'cosine':
            similarities = torch.nn.functional.cosine_similarity(query_embedding, embeddings, dim=1)
        elif similarity_fun_name == 'dot':
            similarities = torch.matmul(query_embedding, embeddings.T).squeeze()
        elif similarity_fun_name == 'euclidean':
            distances = torch.cdist(query_embedding, embeddings, p=2).squeeze()
            similarities = -distances  #convert distances to similarities by negating
        else:
            raise ValueError(f"Unsupported similarity function: {similarity_fun_name}")

        top_k_indices = torch.topk(similarities, top_k).indices

        ranked_list = [doc_ids[idx] for idx in top_k_indices]
        sim_scores = [similarities[idx].item() for idx in top_k_indices]
        return {"ranked_list": ranked_list, "sim_scores": sim_scores}

    def load_individual(self, query_embedding, similarity_fun_name, top_k):
        """
        Finds the top k most similar documents by loading individual embeddings one by one,
        using a priority queue (min-heap).
        """
        results = []
        with open(self.corpus_emb_path, 'rb') as corpus_file:
            try:
                while True:
                    doc = pickle.load(corpus_file)
                    doc_id = doc['doc_id']
                    doc_embedding = doc['embedding']
                    similarity_fun = getattr(torch.nn.functional, similarity_fun_name)
                    similarity = similarity_fun(query_embedding, doc_embedding)
                    if len(results) < top_k:
                        heapq.heappush(results, (similarity, doc_id))
                    else:
                        if similarity > results[0][0]:
                            heapq.heapreplace(results, (similarity, doc_id))
            except EOFError:
                pass

        results = sorted(results, key=lambda x: x[0], reverse=True)
        ranked_list = [doc_id for _, doc_id in results]
        sim_scores = [similarity.item() for similarity, _ in results]
        return {"ranked_list": ranked_list, "sim_scores": sim_scores}

def cosine_similarity(embedding1, embedding2):
    return torch.nn.functional.cosine_similarity(embedding1, embedding2, dim=0).item()

