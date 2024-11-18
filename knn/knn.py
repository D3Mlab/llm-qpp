from abc import ABC, abstractmethod
import pickle
import json
import torch
import heapq

class KNN(ABC):
    def __init__(self, config, corpus_emb_path):
        self.config = config
        self.corpus_emb_path = corpus_emb_path

    @abstractmethod
    def get_top_k(self, query_embedding, similarity_fun, k, implementation = None):
         raise NotImplementedError("This method must be implemented by a subclass.")


class ExactKNN(KNN):

    def get_top_k(self, query_embedding, similarity_fun, k, implementation = matmul_all):
        #read self.config and finds 
        
        return self.implementation(query_embedding, similarity_fun, k)

    def matmul_all(self):
        """
        Loads all embeddings from the corpus and returns them as a list of embeddings.
        """
        embeddings = []
        with open(self.corpus_emb_path, 'rb') as corpus_file:
            try:
                while True:
                    doc = pickle.load(corpus_file)
                    embeddings.append((doc['doc_id'], doc['embedding']))
            except EOFError:
                pass
        return embeddings

    def priority_queue_knn(self, query_embedding, similarity_fun, top_k):
        """
        Finds the top k most similar documents using a priority queue (min-heap).
        """
        results = []
        with open(self.corpus_emb_path, 'rb') as corpus_file:
            try:
                while True:
                    doc = pickle.load(corpus_file)
                    doc_id = doc['doc_id']
                    doc_embedding = doc['embedding']
                    similarity = similarity_fun(query_embedding, doc_embedding)
                    if len(results) < top_k:
                        heapq.heappush(results, (similarity, doc_id))
                    else:
                        if similarity > results[0][0]:
                            heapq.heapreplace(results, (similarity, doc_id))
            except EOFError:
                pass

        results = sorted(results, key=lambda x: x[0], reverse=True)
        return [doc_id for _, doc_id in results]

def cosine_similarity(embedding1, embedding2):
    return torch.nn.functional.cosine_similarity(embedding1, embedding2, dim=0).item()
