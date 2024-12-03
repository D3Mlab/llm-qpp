from eval_manager import EvalManager

if __name__ == "__main__":
    
    eval_manager = EvalManager()
    exp_dir = "experiments/nov_18_setup/trial2_MS_Marco_10K_miniLM_listwise_rr"
    eval_manager.eval_experiment(exp_dir)

    #exp_dir = "experiments/nov_18_setup"
    #eval_manager.eval_experiments(exp_dir)