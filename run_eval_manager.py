from eval_manager import EvalManager

if __name__ == "__main__":
    #DONT FORGET TO ALSO CHANGE THE QRELS PATH IN THE EVAL CONFIG
    exp_dir = "experiments/dec_1_justin/MS-MARCO_q100_miniLM"   
    eval_manager = EvalManager()
    eval_manager.eval_experiments(exp_dir)