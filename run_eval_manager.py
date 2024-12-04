from eval_manager import EvalManager

if __name__ == "__main__":
    eval_manager = EvalManager()
    #DONT FORGET TO ALSO CHANGE THE QRELS PATH IN THE EVAL CONFIG
    exp_dir = "experiments/dec_2/wayfair_5_ali_flash_rerank_k10_T3_no_rep"
    eval_manager.eval_experiment(exp_dir)

    #exp_dir = "experiments/nov_18_setup"
    #eval_manager.eval_experiments(exp_dir)