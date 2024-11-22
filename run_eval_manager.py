from eval_manager import EvalManager

if __name__ == "__main__":
    
    exp_dir = "experiments/nov_18_setup"
    eval_manager = EvalManager()
    eval_manager.eval_experiments(exp_dir)