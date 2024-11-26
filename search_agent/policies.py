
from abc import ABC, abstractclassmethod, abstractmethod
from utils.setup_logging import setup_logging

class BasePolicy(ABC):

    def __init__(self, config):
        
        self.config = config
        self.logger = setup_logging(self.__class__.__name__, self.config)

    @abstractmethod
    def next_action(self, state) -> tuple:
   #return: (<method>, <args>) or None if no next action
         raise NotImplementedError("This method must be implemented by a subclass.")

class PipelinePolicy(BasePolicy):

    def __init__(self, config):
        super().__init__(config)
        from . import COMPONENT_CLASSES, MAIN_ACTIONS        

    def next_action(self, state):



        #e.g. "DenseRetriever"
        first_comp_name =  self.agent_config.get('first_component')
        first_comp_class = COMPONENT_CLASSES.get(first_comp_name)
        first_comp = first_comp_class(config = self.config)

        #get main action - e.g. rank for DenseRetriever, 
        first_comp_main_action = MAIN_ACTIONS.get(first_comp_class)

        self.components = {first_comp_name : {"component" : first_comp, "main_action": types.MethodType(first_comp_main_action, first_comp)}}