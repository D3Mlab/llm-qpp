from experiment_manager import ExperimentManager
import os
from dotenv import load_dotenv
import torch


if __name__ == "__main__":
    # Enable GPU optimizations
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        torch.backends.cudnn.benchmark = True
        #print("Using GPU")
    
    # Load the .env file
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'))
    exp_dir = "experiments/dec_9_new_datasets/rogen_wiki_miniLM_QPP_T3_k30"
    experiment_manager = ExperimentManager()
    experiment_manager.run_experiment(exp_dir)
