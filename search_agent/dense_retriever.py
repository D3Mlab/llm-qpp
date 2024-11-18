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
        query_embedding = self.embedder.embed([query])[0]
        
        #TODO: clean
        knn_method = ExactKNN(self.config, corpus_path=self.config['corpus_path'], similarity_function=self.config['similarity_function'])
        ranked_list = knn_method.rank(query_embedding)
        return {'ranked_list': ranked_list}


        #select ranking method: KNN: superclass -- exactKNN or approxKNN -- make this an interface in utils which can be accesed
        #- needs query embedding
        #- needs corpus path
        #- needs similarity funtion
        #- returns list of top (or aprox) top k docIDs