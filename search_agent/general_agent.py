from utils.setup_logging import setup_logging
from .base_agent import BaseAgent
from . import COMPONENT_CLASSES, MAIN_ACTIONS

class GeneralAgent(BaseAgent):

    def __init__(self, config, data_path_dict):
        #todo - move data paths to agent side
        super().__init__(config, data_path_dict)

    def rank(self, query):

        self.agent_config = self.config.get('agent', {})

        #e.g. "DenseRetriever"
        first_comp_name =  self.agent_config.get('first_component')
        first_comp_class = COMPONENT_CLASSES.get(first_comp_name)
        first_comp = first_comp_class(config = self.config)

        first_comp_main_action = MAIN_ACTIONS.get(first_comp.__class__.mro()[-2])

        self.components = {first_comp_name : {"component" : first_comp, "main_action": first_comp_main_action}}

        #get main action - e.g. rank for DenseRetriever, 
        #main action is defined based on highest level superclass (below Object)


                           #, "main_action" : XX}}

        
        first_action_class = first_action_class_name(config = self.config)

        self.state = {
            'query' : query
            }

