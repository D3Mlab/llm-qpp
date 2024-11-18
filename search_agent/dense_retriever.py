from common.setup_logging import setup_logging
from .base_agent import BaseAgent

class DenseRetriever(BaseAgent):

    def __init__(self, config, data_path_dict):
        super().__init__(config, data_path_dict)

        # Initialize embedder
        self.embedder_config = self.config.get('embedding', {})
        embedder_class = embedding.EMBEDDER_CLASSES.get(self.embedder_config.get('embedder_class'))
        if embedder_class is None:
            raise ValueError("Invalid embedder class specified in config")
        self.embedder = embedder_class(config, model_name=self.embedder_config.get('model_name'))

    def rank(self, query):
    #select ranking method: KNN: superclass -- exactKNN or approxKNN -- make this an interface in utils which can be accesed
        #- needs query embedding
        #- needs corpus path
        #- needs similarity funtion
        #- returns list of top (or aprox) top k docIDs