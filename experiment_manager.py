
import yaml
import json
import argparse
import os
from pathlib import Path
from utils.setup_logging import setup_logging
import search_agent

class ExperimentManager():

    def __init__(self):
        pass

    def run_experiments(self, exp_root: str):
        """
        Given directory, runs experiments for each subdirectory with a 'config.yaml' file.
        """
        self.main_logger = setup_logging(self.__class__.__name__, output_file=os.path.join(exp_root, "experiment.log"))

        for root, dirs, _ in os.walk(exp_root):
            for directory in dirs:
                dir_path = os.path.join(root, directory)
                config_path = os.path.join(dir_path, "config.yaml")
                if os.path.exists(config_path):
                    self.logger.info(f'Running experiment in {dir_path}')
                    self.run_experiment(dir_path)
                else:
                    self.logger.warning(f'No config.yaml in {dir_path}')

    def run_experiment(self, config_dir):
        """
        Executes an experiment based on the 'config.yaml' file in the given directory.
        """

        with open(os.path.join(config_dir, "config.yaml"), "r") as config_file:
            self.config = yaml.safe_load(config_file)

        self.experiment_logger = setup_logging("EXPERIMENT", output_file=os.path.join(config_dir, "experiment.log"), config = self.config)

        #load any previous results (in "detailed_results.json") as dictionary
        results = self.load_past_results(config_dir)
        #results = {
        #    query_index: {
        #        'ranked_list': [<list of ranked docIDs>]
        #    },
        #    ...
        #  }

        #get dictionary of data paths:
        data_path_dict = self.setup_data_paths(self.config)
        #e.g. data_path_dict = 
            #{"embeddings_path": "emb.pkl", "text_path": "collection.jsonl", ...}

        self.experiment_logger.debug(f'data paths: {data_path_dict}')

        

    def setup_data_paths(self, config):
        """
        Read config to see which keys need to be included in the data_path_dict. Empty keys will not be included.
        """
        data_paths_config = config.get('data_paths', {})
        data_path_dict = {key: path for key, path in data_paths_config.items() if isinstance(path, str) and path.strip()}
        return data_path_dict

    def load_past_results(self, config_dir):
        detailed_results_path = os.path.join(config_dir, 'detailed_results.json')
        if os.path.exists(detailed_results_path):
            with open(detailed_results_path, 'r') as file:
                return json.load(file)
        return {}