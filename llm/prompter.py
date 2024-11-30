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
        self.template_config = config.get("templates", {})

        model_class = LLM_CLASSES.get(self.model_class_name)
        self.llm = model_class(config,self.model_name)

        self.jinja_env = jinja2.Environment(loader=jinja2.FileSystemLoader(searchpath=self.template_dir))

    def reform_q_uninformed(self, state):
        #reformulate initial query (pre-retrieval)

        init_q = self.get_init_q(state)

        prompt_dict = {"query" : init_q}

        template_dir = self.template_config["uninformed_query_reformulation"]
        prompt = self.render_prompt(prompt_dict, template_dir)

        reformed_q = self.llm.prompt(prompt)["message"]

        state["queries"].append(reformed_q)
        return state

    def rerank_best_and_latest(self, state):
        #given a) the previously best list of length K and b) K additional retrieved docs
        #rerank these 2xK docs to a list of length K

        #if previously best list is empty:
        #

        pass

    def get_init_q(self,state):
        return state["queries"][0]

    def render_prompt(self, prompt_dict, template_dir):
        template = self.jinja_env.get_template(template_dir)
        return template.render(prompt_dict)

#E.g. jinja2 macro usage:
# In "macros.jinja"
#{% macro greet(name) %}
#  Hello, {{ name }}!
#{% endmacro %}

#{% import "macros.jinja" as macros %}
#{{ macros.greet("FIRST NAME") }}


#if __name__ == __main__:

#    from . import LLM_CLASSES
 #   import llms



