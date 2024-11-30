
import os
import yaml
from dotenv import load_dotenv
from . import LLM_CLASSES
from .prompter import Prompter

#tester code for llms
#should run as module: >python -m llm
if __name__ == "__main__":

    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    with open(config_path, 'r') as config_file:
        config = yaml.safe_load(config_file)

    load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

    prompter = Prompter(config)

    init_state = {"queries": ["q^0", "q^1"]}

    print('init_state: ', init_state)

    new_state = prompter.reform_q_uninformed(init_state)

    print('new_state: ', new_state)

    #llm_class = LLM_CLASSES.get(model_class)

    #llm = llm_class(config, model_name)
    #prompt = "2+2="
    #response = llm.prompt(prompt)

    #print(response)
