from experiment_manager import ExperimentManager
import os
from dotenv import load_dotenv

if __name__ == "__main__":
    
    # Load the .env file
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'))

    exp_dir = "experiments/dec_2/esci_100_ali_k10"
    experiment_manager = ExperimentManager()
    experiment_manager.run_experiment(exp_dir)
