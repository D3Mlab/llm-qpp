from experiment_manager import ExperimentManager

if __name__ == "__main__":
    
    exp_dir = "experiments/nov_18_setup/trial3_general_agent"
    experiment_manager = ExperimentManager()
    experiment_manager.run_experiment(exp_dir)
