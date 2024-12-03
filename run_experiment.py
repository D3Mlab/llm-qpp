from experiment_manager import ExperimentManager
import os
from dotenv import load_dotenv

if __name__ == "__main__":
    
    # Load the .env file
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'))

    exp_dir = "experiments/dec_2/wayfair_5_ali_flash_rerank_k10_T3"
    experiment_manager = ExperimentManager()
    experiment_manager.run_experiment(exp_dir)
