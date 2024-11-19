from common.setup_logging import setup_logging
from .base_agent import BaseAgent
import knn

class DenseRetriever(BaseAgent):

    def __init__(self, config, data_path_dict):
        super().__init__(config, data_path_dict)

        # Initialize embedder
        self.embedder_config = self.config.get('embedding', {})
        embedder_class = embedding.EMBEDDER_CLASSES.get(self.embedder_config.get('embedder_class'))
        self.embedder = embedder_class(config, model_name=self.embedder_config.get('model_name'))

        # Initialize KNN
        self.knn_config = self.config.get('knn', {})
        knn_class = knn.KNN_CLASSES.get(self.knn_config.get('knn_class'))
        self.knn = knn_class(config, data_path_dict["emb_path"])

    def rank(self, query):
        # Embed query
        query_embedding = self.embedder.embed([query])[0].to(dtype=torch.float32, device='cuda' if torch.cuda.is_available() else 'cpu')

        #todo: move query to device
        retriever_result = {"query_embedding" : query_embedding}

        #read knn implementation \in {load_all, load_iteratively}
        knn_implmentation = self.knn_config.get('implementation')
        sim_f = self.knn_config.get('sim_f')
        k = self.knn_config.get('k')
        knn_result = self.knn.get_top_k(query_embedding,sim_f,k,knn_implmentation)

        retriever_result.update(knn_result)
        #retriever_result = {"ranked_list": <docID list>, "sim_scores": <list of sim scores>}
        return retriever_result