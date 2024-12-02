from experiment_manager import ExperimentManager
import os
from dotenv import load_dotenv

if __name__ == "__main__":
    
    # Load the .env file
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'))

    exp_dir = "experiments/nov_18_setup/trial3_general_agent"
    experiment_manager = ExperimentManager()
    experiment_manager.run_experiment(exp_dir)
