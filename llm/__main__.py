
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

    init_state = {"queries": ["email your doctor from Hillcrest Clinics", "email your doctor contact details Hillcrest Clinics"], "curr_top_k_docIDs": ["2506722","1331194"], "retrieved_lists" : [["1331194", "2697809"],["1331194", "2697809"]]}

    new_state = prompter.rerank_best_and_latest(init_state)

    print('new_state: ', new_state)



    #TO RUN LLM:

    #llm_class = LLM_CLASSES.get(model_class)

    #llm = llm_class(config, model_name)
    #prompt = "2+2="
    #response = llm.prompt(prompt)

    #print(response)
