import yaml
import json
import os
from pathlib import Path
from utils.setup_logging import setup_logging
import pytrec_eval
import numpy as np
from scipy.stats import norm


# EvalManager class for evaluation
class EvalManager():

    def __init__(self, ):

        pass

    def eval_experiments(self, exp_root: str):
        """
        Evaluates all experiments in the given root directory.
        """
        self.exp_root = Path(exp_root)

        self.main_logger = setup_logging(self.__class__.__name__, output_file=self.exp_root / "eval.log")

        for root, dirs, _ in os.walk(self.exp_root):
            root_path = Path(root)
            if (root_path / "config.yaml").exists() and "per_query_results" in dirs:
                self.main_logger.info(f"Evaluating experiment in {root_path.as_posix()}")
                self.eval_experiment(root_path)

    def eval_experiment(self, experiment_dir):
        """
        Evaluates a single experiment directory.
        """
        experiment_dir = Path(experiment_dir)

        eval_config_path = Path(experiment_dir)  / "eval_config.yaml"
        
        if eval_config_path.exists():
            with open(eval_config_path, "r") as config_file:
                self.config = yaml.safe_load(config_file)
        else:
            raise FileNotFoundError(f"Could not find {eval_config_path}")

        self.experiment_logger = setup_logging(f"EVALUATOR_{experiment_dir} ", output_file= experiment_dir / "eval.log", config = self.config)
       
        selected_measures = self.config.get("measures", pytrec_eval.supported_measures)
        #todo confirm selected_measures structure
        conf_level = float(self.config.get("confidence_level", 0.95))
        
        results_dir = Path(experiment_dir) / 'per_query_results'

        self.all_query_eval_results = {}

        for query_dir in results_dir.iterdir():
            print("QUERY DIR", query_dir)
            if query_dir.is_dir():
                self.evaluate_single_query(query_dir, selected_measures)

        # Compute mean and confidence interval for each measure
        self.write_all_queries_eval_results(experiment_dir, selected_measures, conf_level)

    def evaluate_single_query(self, query_dir, selected_measures):
        """
        Evaluates a single query within an experiment directory.
        """
        trec_file_path = query_dir / "trec_results_raw.txt"
        dedup_trec_file_path = query_dir / "trec_results_deduplicated.txt"
        eval_results_path = query_dir / "eval_results.jsonl"

        # Remove duplicates from TREC file and save deduplicated version
        #todo: check deduplication behaviour
        #todo: add count of deduplicated documents to saved results for the query (similar to qpp results)
        deduped_lines = self.deduplicate_trec_results(trec_file_path, dedup_trec_file_path)
        

        # Parse deduplicated TREC results
        results = pytrec_eval.parse_run(deduped_lines)
        print("RESULTS", results)

        # Evaluate using pytrec_eval
        evaluator = pytrec_eval.RelevanceEvaluator(self.load_qrels(), selected_measures)
        per_query_eval_results = evaluator.evaluate(results)
        print("PER QUERY EVAL RESULTS", per_query_eval_results)

        # Write per-query evaluation results to JSONL
        self.write_jsonl(eval_results_path, per_query_eval_results)

        # Store results for calculating averages and confidence intervals
        print("QUERY DIR NAME", query_dir.name)
        print("PER QUERY EVAL RESULTS", per_query_eval_results)
        self.all_query_eval_results[query_dir.name] = per_query_eval_results[query_dir.name]

    def load_qrels(self):
        """
        Loads the QRELs from the path specified in the evaluation config.
        """
        qrels_path = self.config.get("qrels_path")
        with open(qrels_path, "r") as qrels_file:
            return pytrec_eval.parse_qrel(qrels_file)

    def deduplicate_trec_results(self, trec_file_path, dedup_trec_file_path):
        """
        Removes duplicate documents from a TREC results file and saves deduplicated version.
        """
        if not trec_file_path.exists() or trec_file_path.stat().st_size == 0:
            query_id = trec_file_path.parent.name  # Use the parent directory name as the query ID
            self.experiment_logger.warning(f"Query {query_id}: TREC results file {trec_file_path} is empty or missing. Adding a dummy line.")
            dummy_line = f"{query_id} Q0 dummy_doc_id 1 0.0 dummy_run\n"
            with open(dedup_trec_file_path, "w") as dedup_file:
                dedup_file.write(dummy_line)
            return [dummy_line]
        
        with open(trec_file_path, "r") as trec_file:
            lines = trec_file.readlines()
            seen_docs = set()
            deduped_lines = []
            for line in lines:
                doc_id = line.split()[2]
                if doc_id not in seen_docs:
                    deduped_lines.append(line)
                    seen_docs.add(doc_id)

        # Write deduplicated lines to file
        with open(dedup_trec_file_path, "w") as dedup_file:
            dedup_file.writelines(deduped_lines)

        return deduped_lines

    def write_jsonl(self, file_path, data):
        """
        Writes dictionary data to a JSONL file.
        """
        with open(file_path, "w") as file:
            for query_id, eval_result in data.items():
                json.dump({"query_id": query_id, **eval_result}, file)
                file.write("\n")


    def write_all_queries_eval_results(self, experiment_dir, selected_measures, conf_level):
        """
        Writes the mean and confidence interval results for all queries in an experiment.
        """
        mean_results = {}
        ci_results = {}

        self.experiment_logger.debug(f"self.all_query_eval_results {self.all_query_eval_results} ")

        for measure in selected_measures:
            values = [result.get(measure) for result in self.all_query_eval_results.values() if result.get(measure) is not None]
            self.experiment_logger.debug(f"Values for measure {measure}: {values}")
            if values:
                mean_value = np.mean(values)
                std_dev = np.std(values, ddof=1)
                z = norm.ppf((1 + conf_level) / 2)
                margin_of_error = z * std_dev / np.sqrt(len(values))

                mean_results[f"mean_{measure}"] = mean_value
                ci_results[f"mean_{measure}_ub_ci"] = mean_value + margin_of_error
                ci_results[f"mean_{measure}_lb_ci"] = mean_value - margin_of_error

        all_eval_results = {**mean_results, **ci_results}
        all_eval_results_path = Path(experiment_dir) / "all_queries_eval_results.jsonl"
        with open(all_eval_results_path, "w") as file:
            self.experiment_logger.debug(f"Writing final evaluation results: {all_eval_results}")
            json.dump(all_eval_results, file)
            file.write("\n")