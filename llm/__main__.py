
import os
import yaml
from dotenv import load_dotenv
from . import LLM_CLASSES

#tester code for llms
#should run as module: >python -m llm
if __name__ == "__main__":
    # Load configuration from config.yaml
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
    with open(config_path, 'r') as config_file:
        config = yaml.safe_load(config_file)

    # Extract model family and model name from the configuration
    llm_config = config.get('llm', {})
    model_family = llm_config.get('model_family')
    model_name = llm_config.get('model_name')

    # Load the .env file
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

    # Get the required LLM class from LLM_CLASSES
    llm_class = LLM_CLASSES.get(model_family)

    # Instantiate the LLM class and test the prompt method
    llm = llm_class(config, model_name)
    prompt = "2+2="
    response = llm.prompt(prompt)

    # Print the response
    print(response)
