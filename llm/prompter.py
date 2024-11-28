import configparser
from utils.setup_logging import setup_logging
from . import LLM_CLASSES

class Prompter():
    
    def __init__(self,config):

        self.config = config 
        self.logger = setup_logging(self.__class__.__name__, self.config)

        self.llm_config = config.get('llm', {})
        self.model_class_name = self.llm_config.get('model_class')
        self.model_name = self.llm_config.get('model_name')

        model_class = LLM_CLASSES.get(self.model_class_name)
        self.llm = model_class(config,self.model_name)

        prompt = "2+2="
        response = self.llm.prompt(prompt)

        print(response)




