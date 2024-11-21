
import random
from typing import Optional
import yaml
import json
import argparse
import os
from pathlib import Path
from common.setup_logging import setup_logging
import data_loaders
from data_loaders.data_loader import DataLoader
import ranking
from ranking.ranker import Ranker
import numpy as np
import csv

class ExperimentManager():
    
    def __init__(self):
        self.logger = setup_logging(self.__class__.__name__, {'level': 'INFO'})
        self.experiment_logger = None

    def run_experiments(self, exp_dir: str):
        """
        Runs experiments for each subdirectory in the provided directory path.
        """
        for root, dirs, _ in os.walk(exp_dir):
            for directory in dirs:
                config_file_path = os.path.join(root, directory, "config.yaml")
                dir_path = os.path.join(root, directory)
                self.run_experiment(config_file_path, dir_path=dir_path)

    def run_experiment(self, config_path, dir_path=''):
        """
        Executes an experiment based on the provided configuration file.
        """
        self.load_config(config_path)
        self.setup_experiment_logger(dir_path)
        self.experiment_logger.info(f'running experiment on directory: {dir_path}')

        self.data_loader = self.initialize_data_loader()
        self.ranker = self.initialize_ranker()
        queries = self.get_queries()

        results_dir = Path(dir_path) / 'per_query_results'
        results_dir.mkdir(parents=True, exist_ok=True)

        existing_results = self.load_existing_results(results_dir)
        self.rank_queries(queries, results_dir, existing_results)

    def load_config(self, config_path):
        """
        Loads the experiment configuration from a YAML file.
        """
        try:
            with open(config_path, "r") as config_file:
                self.config = yaml.safe_load(config_file)
        except Exception as e:
            self.logger.error(f"Failed to load config file {config_path}: {e}")
            raise

    def setup_experiment_logger(self, log_directory):
        """
        Setup the logger for the experiment.
        """
        self.config['logging']['log_directory'] = log_directory
        self.experiment_logger = setup_logging('EXPERIMENT', self.config)
        self.logger = setup_logging(self.__class__.__name__, self.config)

    def initialize_data_loader(self):
        """
        Initializes the data loader based on the configuration.
        """
        data_loader_class = data_loaders.DATA_LOADER_CLASSES[self.config['data']['data_loader_name']]
        return data_loader_class(self.config)

    def initialize_ranker(self):
        """
        Initializes the ranker based on the configuration.
        """
        ranker_class = ranking.RANKER_CLASSES[self.config['ranking']['ranker_name']]
        return ranker_class(self.config, self.data_loader)

    def get_queries(self):
        """
        Retrieves queries from a .tsv file based on the configuration.
        """
        queries_path = self.config['data_paths'].get('queries_path')
        queries_dict = {}
        try:
            with open(queries_path, 'r') as file:
                tsv_reader = csv.reader(file, delimiter='\t')
                for row in tsv_reader:
                    qid, query_text = row
                    queries_dict[qid] = query_text
        except Exception as e:
            self.experiment_logger.error(f"Failed to read queries from {queries_path}: {e}")
            raise RuntimeError(f"Failed to read queries from {queries_path}")

        return queries_dict

    def load_existing_results(self, results_dir):
        """
        Loads existing results by checking for existing result files in the results directory.
        """
        existing_results = set()
        for root, _, files in os.walk(results_dir):
            for file in files:
                if file.endswith("_detailed_results.json"):
                    qid = file.split("_")[0]  # Extract qid from filename like '<qid>_detailed_results.json'
                    existing_results.add(qid)
        return existing_results

    def rank_queries(self, queries, results_dir, existing_results):
        """
        Ranks the provided queries and saves results per query.
        """
        for qid, query in queries.items():
            if qid in existing_results:
                self.experiment_logger.info(f'Results already available for query: {qid} {query}')
                continue

            query_result_dir = results_dir / f"{qid}"
            query_result_dir.mkdir(parents=True, exist_ok=True)
            detailed_results_path = query_result_dir / f"{qid}_detailed_results.json"
            trec_file_path = query_result_dir / f"{qid}_trec_results.txt"

            self.experiment_logger.info(f'Ranking query: {qid} {query}')
            try:
                result = self.ranker.rank(query)
                self.experiment_logger.info('Rank successful')
                self.write_query_result(detailed_results_path, trec_file_path, qid, query, result)
            except Exception as e:
                self.experiment_logger.error(f'Failed to rank query {qid}: {str(e)}')

    def write_query_result(self, detailed_results_path, trec_file_path, qid, query, result):
        """
        Writes the result for a single query to a JSON file and saves TREC results.
        """
        # Write detailed results to JSON file
        with open(detailed_results_path, 'w') as file:
            json.dump(result, file, default=lambda x: x.tolist() if isinstance(x, np.float32) else x, indent=4)

        # Write TREC results
        trec_results = []
        ranked_list = result.get('ranked_list', [])
        for doc_index, doc_id in enumerate(ranked_list):
            score = len(ranked_list) - doc_index
            trec_results.append(f"{qid} Q0 {doc_id} {doc_index + 1} {score} my_run")

        with open(trec_file_path, "w") as trec_file:
            trec_file.write("\n".join(trec_results))

if __name__ == "__main__":
    experiment_manager = ExperimentManager()
    parser = argparse.ArgumentParser()
    parser.add_argument("-exp_dir", "--experiment_dir", type=str)
    args = parser.parse_args()
    experiment_manager.run_experiments(args.experiment_dir)
