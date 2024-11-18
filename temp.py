#canvas nov 18 - 1:49

import random
from typing import Optional
import yaml
import json
import argparse
import os
import numpy as np
import time
import logging

class ExperimentManager():

    def __init__(self):
        pass

    def run_experiments(self, exp_dir: str):
        """
        Given directory, runs experiments for each subdirectory with a 'config.yaml' file.
        """
        self.main_logger = setup_logging(self.__class__.__name__, output_file=exp_dir)

        for root, dirs, _ in os.walk(exp_dir):
            for directory in dirs:
                dir_path = os.path.join(root, directory)
                config_path = os.path.join(dir_path, "config.yaml")
                if os.path.exists(config_path):
                    self.main_logger.info(f'Running experiment in {dir_path}')
                    self.run_experiment(dir_path)
                else:
                    self.main_logger.warning(f'No config.yaml in {dir_path}')

    def run_experiment(self, dir_path):
        """
        Executes an experiment based on the 'config.yaml' file in the given directory.
        """
        with open(os.path.join(dir_path, "config.yaml"), "r") as config_file:
            self.config = yaml.safe_load(config_file)

        self.experiment_logger = setup_logging("EXPERIMENT", output_file=dir_path, config=self.config)

        # Load any previous results (in "detailed_results.json") as a dictionary
        results = self.load_results(dir_path)

        # Get dictionary of data paths (it will be an argument to the ranker initialization)
        data_path_dict = self.setup_data_paths(self.config)
        # e.g., data_path_dict = {"embeddings_path": "/full/path/emb.pkl", "text_path": "/full/path/collection.jsonl"}

        # Setup and run queries
        queries = self.load_queries()
        self.run_queries(queries, results, dir_path)
        self.save_results_in_trec_format(results, dir_path)

    def setup_data_paths(self, config):
        """
        Read config to see which keys need to be included in the data_path_dict.
        E.g., YAML block:
        data_paths: 
          text: /full/path/collection.jsonl
          embeddings: /full/path/emb.pkl
        
        Returns the data path dict.
        """
        data_paths_config = config.get('data_paths', {})
        data_path_dict = {key: path for key, path in data_paths_config.items() if isinstance(path, str) and path.strip()}
        return data_path_dict

    def load_results(self, dir_path):
        detailed_results_path = os.path.join(dir_path, 'detailed_results.json')
        if os.path.exists(detailed_results_path):
            with open(detailed_results_path, 'r') as file:
                return json.load(file)
        return {}

    def load_queries(self):
        """
        Load queries based on the experiment configuration.
        """
        query_read_function = self.config['data'].get('query_read_function', 'get_queries')
        queries = getattr(self, query_read_function)()

        num_queries = self.config['experiment_settings'].get('num_queries')
        if num_queries is not None:
            queries = random.sample(queries, num_queries)
        return queries

    def run_queries(self, queries, results, dir_path):
        retry_delay = self.config['llm']['retry_delay']
        max_retries = self.config['experiment_settings']['max_retries']

        for qid, query in queries:
            if qid in results and results[qid] is not None:
                self.experiment_logger.info(f"results already available for query: {qid} {query}")
                continue

            self.experiment_logger.info(f"calling rank(query) for query: {qid} {query}")
            attempt = 0

            while True:
                try:
                    results[qid] = self.rank(query)
                    if 'ranked_list_full' in results[qid]:
                        del results[qid]['ranked_list_full']
                    self.experiment_logger.info('rank successful')
                    break
                except Exception as e:
                    self.experiment_logger.info(f'{str(e)}, trying again in {retry_delay} s')
                    time.sleep(retry_delay)
                    attempt += 1
                    if attempt == max_retries:
                        self.experiment_logger.warning(f"WARNING: query {qid} not ranked.")
                        results[qid] = None
                        break

            with open(os.path.join(dir_path, 'detailed_results.json'), 'w') as file:
                json.dump(results, file, default=lambda x: x.tolist() if isinstance(x, np.float32) else x, indent=4)

    def save_results_in_trec_format(self, results, dir_path):
        trec_results = []
        k_i = self.config['ranking']['k_i']
        for qid, result in results.items():
            if result is not None:
                ranked_list = result['ranked_list']
                for doc_index, doc_id in enumerate(ranked_list):
                    score = len(ranked_list) - doc_index
                    trec_results.append(f"{qid} Q0 {doc_id} {doc_index + 1} {score} my_run")
            else:
                trec_results.extend(self.get_null_trec(qid, k_i))

        trec_file_path = os.path.join(dir_path, "trec_results.txt")
        with open(trec_file_path, "w") as trec_file:
            trec_file.write("\n".join(trec_results))

    def get_null_trec(self, qid, k_i):
        placeholder_results = []
        for doc_index in range(k_i):
            placeholder_doc_id = f"null_doc{doc_index}"
            score = k_i - doc_index
            placeholder_results.append(f"{qid} Q0 {placeholder_doc_id} {doc_index + 1} {score} my_run")
        return placeholder_results

    def rank(self, query):
        """
        Placeholder rank function. In the actual implementation, this should call the ranking model.
        """
        # Placeholder: return a mock ranking result
        return {'ranked_list': [f'doc_{i}' for i in range(10)]}

if __name__ == "__main__":
    experiment_manager = ExperimentManager()
    parser = argparse.ArgumentParser()
    parser.add_argument("-exp_dir", "--experiment_dir", type=str)
    args = parser.parse_args()
    experiment_manager.run_experiments(args.experiment_dir)
