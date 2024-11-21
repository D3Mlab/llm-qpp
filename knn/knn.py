from abc import ABC, abstractmethod
from utils.setup_logging import setup_logging
import pickle
import json
import torch
import heapq

class KNN(ABC):
    def __init__(self, config, corpus_emb_path):
        self.config = config
        self.logger = setup_logging(self.__class__.__name__, self.config)
        self.corpus_emb_path = corpus_emb_path
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.logger.debug("Initialized KNN")

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
        and returns the top_k most similar document IDs.
        """
        docs = []
        with open(self.corpus_emb_path, 'rb') as corpus_file:
            try:
                while True:
                    # Load individual documents from the file as they are saved separately
                    doc = pickle.load(corpus_file)
                    docs.append(doc)
            except EOFError:
                pass



        doc_ids = [doc['doc_id'] for doc in docs]
        embeddings = torch.stack([doc['embedding'] for doc in docs]).to(dtype=torch.float32, device=self.device)

        # Compute similarity based on the selected function
        if similarity_fun_name == 'cosine':
            similarities = torch.nn.functional.cosine_similarity(query_embedding.unsqueeze(0), embeddings, dim=1)
        elif similarity_fun_name == 'dot':
            similarities = torch.matmul(embeddings,query_embedding).squeeze()
        elif similarity_fun_name == 'euclidean':
            distances = torch.cdist(query_embedding.unsqueeze(0), embeddings, p=2).squeeze()
            similarities = -distances  #convert distances to similarities by negating
        else:
            raise ValueError(f"Unsupported similarity function: {similarity_fun_name}")

        top_k_indices = torch.topk(similarities, top_k).indices

        ranked_list = [doc_ids[idx] for idx in top_k_indices]
        sim_scores = similarities[top_k_indices].tolist()
        return {"ranked_list": ranked_list, "sim_scores": sim_scores}

    def load_individual(self, query_embedding, similarity_fun_name, top_k):
        #this generated code for using a priority queue to avoid OOME is unchecked/untested
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


class ApproxKNN(KNN):
    #TODO (e.g. using FAISS)
    def __init__(self, config, corpus_emb_path):
        super().__init__(config, corpus_emb_path)
        pass
