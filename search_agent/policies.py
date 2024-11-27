
from abc import ABC, abstractclassmethod, abstractmethod
#from search_agent import COMPONENT_CLASSES
from utils.setup_logging import setup_logging

class BasePolicy(ABC):

    def __init__(self, config):
        
        self.config = config
        self.logger = setup_logging(self.__class__.__name__, self.config)

        from . import COMPONENT_CLASSES 
        self.COMPONENT_CLASSES = COMPONENT_CLASSES

    @abstractmethod
    def next_action(self, state) -> tuple:
   #return: (<method>, <args>) or None if no next action
         raise NotImplementedError("This method must be implemented by a subclass.")

class PipelinePolicy(BasePolicy):
    
    def __init__(self, config):
        super().__init__(config)
        self.components = {}
        self.steps = config.get('agent', {}).get('policy_steps', [])
        self.current_step = 0

    def next_action(self, state):
        if self.current_step >= len(self.steps):
            return None

        step = self.steps[self.current_step]
        #class which takes the step e.g. DenseRetriever, Reranker
        comp_name = step.get('component')
        #method to take, e.g. rank(query), rerank(query, ranked_list)
        method_name = step.get('method')

        self.logger.debug(f"Policy step {step}")

        #check if we've already instantiated the component (e.g the dense retriever)
        if comp_name not in self.components:
            #instantiate new component
            comp_class = self.COMPONENT_CLASSES.get(comp_name)
            if not comp_class:
                raise ValueError(f"Component class for {comp_name} not found.")
            comp_inst = comp_class(config=self.config)
            self.components[comp_name] = comp_inst
        else:
            comp_inst = self.comonents[comp_name]

        # Get the method from the component instance, e.g. rank() from DenseRetriever
        act_method = getattr(comp_inst, method_name)

        self.current_step += 1
        return act_method

