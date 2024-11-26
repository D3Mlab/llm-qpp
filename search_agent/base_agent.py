from abc import ABC, abstractclassmethod, abstractmethod
from utils.setup_logging import setup_logging

class BaseAgent(ABC):

    def __init__(self, config):
        
        self.config = config
        self.agent_config = self.config.get('agent', {})
        self.logger = setup_logging(self.__class__.__name__, self.config)

        data_paths_config = self.config.get('data_paths', {})
        self.data_path_dict = {key: path for key, path in data_paths_config.items() if isinstance(path, str) and path.strip()}


    @abstractmethod
    def rank(self, query: str) -> dict:
   #return:
   #results = { 
        #        'ranked_list': [<list of ranked docIDs>]
        #    ...
        #  }
         raise NotImplementedError("This method must be implemented by a subclass.")
