import configparser
from utils.setup_logging import setup_logging
from . import LLM_CLASSES

import jinja2

class Prompter():
    
    def __init__(self,config):

        self.config = config 
        self.logger = setup_logging(self.__class__.__name__, self.config)

        self.llm_config = config.get('llm', {})
        self.model_class_name = self.llm_config.get('model_class')
        self.model_name = self.llm_config.get('model_name')
        self.template_dir = self.llm_config.get('template_dir')

        model_class = LLM_CLASSES.get(self.model_class_name)
        self.llm = model_class(config,self.model_name)

        self.jinja_env = jinja2.Environment(loader=jinja2.FileSystemLoader(searchpath=self.template_dir))

    def reform_q_uninformed(self, state):
        #args:  state dictionary with a {"queries" [q^0, ..., q^T] where q^0 is initial query and q^T is the most recent query reformulation
        #reformulate only initial query q^0



        pass       


if __name__ == __main__:

    from . import LLM_CLASSES
    import llms



