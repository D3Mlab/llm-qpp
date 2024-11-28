from utils.setup_logging import setup_logging
from .base_agent import BaseAgent
import torch
import embedding
import knn
import copy

class DenseRetriever(BaseAgent):

    def __init__(self, config):
        super().__init__(config)

        # Initialize embedder
        self.embedder_config = self.config.get('embedding', {})
        embedder_class = embedding.EMBEDDER_CLASSES.get(self.embedder_config.get('embedder_class'))
        self.embedder = embedder_class(config = config, model_name=self.embedder_config.get('model_name'))

        # Initialize KNN
        self.knn_config = self.config.get('knn', {})
        knn_class = knn.KNN_CLASSES.get(self.knn_config.get('knn_class'))
        self.knn = knn_class(config, self.data_path_dict["emb_path"])

        self.logger.debug("Initialized Dense Retreiver")

    def rank(self, state):
        
        #start building new state
        retriever_result = copy.deepcopy(state)

        query = state["query"]

        # Embed query
        query_embedding = self.embedder.embed([query])[0].to(dtype=torch.float32)
        #start building result dictionary
        retriever_result = {"query_embedding" : query_embedding}

        #read knn implementation \in {load_all, load_iteratively}
        knn_implmentation = self.knn_config.get('implementation')
        sim_f = self.knn_config.get('sim_f')
        k = self.knn_config.get('k')
        knn_result = self.knn.get_top_k(query_embedding,sim_f,k,knn_implmentation)

        retriever_result.update(knn_result)
        #retriever_result = {"ranked_list": <docID list>, 
        #                   "sim_scores": <list of sim scores>, 
        #                   "query_embedding" : query_embedding,
        #                   "query" : <q> }
        return retriever_result