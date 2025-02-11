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
            'load_all_in_batches': self.load_all_in_batches
        }

    def get_top_k(self, query_embedding, similarity_fun_name, k, implementation, prev_retrieved):
        implementation_method = self.IMPLEMENTATION_METHODS.get(implementation,'load_all')
        query_embedding = query_embedding.to(self.device)
        return implementation_method(query_embedding, similarity_fun_name, k, prev_retrieved)

    #todo: check batch ranking gives same ranking as monolithic on 100 docs with batch 10
    #todo: remove pytorch warnings, clean up redundant code for similarity
    def load_all_in_batches(self, query_embedding, similarity_fun_name, top_k, prev_retrieved, batch_size=100):
        """
        Loads embeddings in batches, computes similarity with the query embedding,
        and returns the top_k most similar document IDs.
        """
        query_embedding = query_embedding.to(self.device)
        doc_ids = []
        similarities = []

        with open(self.corpus_emb_path, 'rb') as corpus_file:
            try:
                batch_embeddings = []
                batch_doc_ids = []
                while True:
                    doc = pickle.load(corpus_file)
                    batch_embeddings.append(torch.tensor(doc['embedding'], dtype=torch.float32)
                        if not isinstance(doc['embedding'], torch.Tensor)
                        else doc['embedding'].clone().detach().to(dtype=torch.float32))
                    batch_doc_ids.append(doc['doc_id'])

                    # Process a batch if it reaches batch_size
                    if len(batch_embeddings) == batch_size:
                        # Stack and move batch to the device
                        batch_embeddings = torch.stack(batch_embeddings).to(self.device)

                        # Compute similarity for the current batch
                        batch_similarities = self.compute_similarity(query_embedding, batch_embeddings, similarity_fun_name)

                        # Collect results
                        doc_ids.extend(batch_doc_ids)
                        similarities.extend(batch_similarities.tolist())

                        # Clear batch
                        batch_embeddings = []
                        batch_doc_ids = []

            except EOFError:
                pass

            # Process any remaining embeddings in the last batch
            if batch_embeddings:
                batch_embeddings = torch.stack(batch_embeddings).to(self.device)
                batch_similarities = self.compute_similarity(query_embedding, batch_embeddings, similarity_fun_name)

                # Collect results for the last batch
                doc_ids.extend(batch_doc_ids)
                similarities.extend(batch_similarities.tolist())

        return self.process_results(similarities, doc_ids, prev_retrieved, top_k)


    def load_all(self, query_embedding, similarity_fun_name, top_k, prev_retrieved):
        #todo - refactor to use similarity function and filtering function...
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
        similarities = self.compute_similarity(query_embedding, embeddings, similarity_fun_name).tolist()

        return self.process_results(similarities, doc_ids, prev_retrieved, top_k)

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

    def compute_similarity(self, query_embedding, embeddings, similarity_fun_name):
        """
        Computes similarity between the query embedding and batch embeddings.
        """
        if similarity_fun_name == 'cosine':
            return torch.nn.functional.cosine_similarity(query_embedding.unsqueeze(0), embeddings, dim=1)
        elif similarity_fun_name == 'dot':
            return torch.matmul(embeddings, query_embedding).squeeze()
        elif similarity_fun_name == 'euclidean':
            distances = torch.cdist(query_embedding.unsqueeze(0), embeddings, p=2).squeeze()
            return -distances  # convert distances to similarities by negating
        else:
            raise ValueError(f"Unsupported similarity function: {similarity_fun_name}")

    def process_results(self, similarities, doc_ids, prev_retrieved, top_k):
        """
        Processes raw similarities and document IDs to generate a filtered and ranked list.
    
        Args:
            similarities (list or tensor): List or tensor of similarity scores.
            doc_ids (list): List of document IDs corresponding to the similarities.
            prev_retrieved (set): Set of previously retrieved document IDs.
            top_k (int): Number of top results to return.

        Returns:
         dict: A dictionary containing:
                - "ranked_list": List of top_k document IDs.
                - "sim_scores": List of corresponding similarity scores.
        """
        # Sort by similarity in descending order and filter out prev_retrieved
        results = sorted(zip(similarities, doc_ids), key=lambda x: x[0], reverse=True)
        filtered_results = [(sim, doc_id) for sim, doc_id in results if doc_id not in prev_retrieved]

        # Truncate to top_k
        filtered_results = filtered_results[:top_k]

        # Separate back into ranked list and similarity scores
        ranked_list = [doc_id for _, doc_id in filtered_results]
        sim_scores = [similarity for similarity, _ in filtered_results]

        return {"ranked_list": ranked_list, "sim_scores": sim_scores}

class ApproxKNN(KNN):
    #TODO (e.g. using FAISS)
    def __init__(self, config, corpus_emb_path):
        super().__init__(config, corpus_emb_path)
        pass
