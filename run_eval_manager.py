from eval_manager import EvalManager

if __name__ == "__main__":

    eval_manager = EvalManager()
    exp_dir = "experiments/dec_2/wayfair_20_ali_k10"

    eval_manager.eval_experiment(exp_dir)

    #exp_dir = "experiments/nov_18_setup"
    #eval_manager.eval_experiments(exp_dir)