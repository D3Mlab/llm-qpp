from abc import ABC, abstractclassmethod, abstractmethod
from utils.setup_logging import setup_logging

class BaseAgent(ABC):

    def __init__(self, config, data_path_dict):
        
        self.config = config
        self.logger = setup_logging(self.__class__.__name__, self.config)
        self.data_path_dict = data_path_dict

    @abstractmethod
    def rank(self, query: str) -> dict:
   #return:
   #results = { 
        #        'ranked_list': [<list of ranked docIDs>]
        #    ...
        #  }
         raise NotImplementedError("This method must be implemented by a subclass.")
