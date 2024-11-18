
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
                    self.logger.info(f'Running experiment in {dir_path}')
                    self.run_experiment(dir_path)
                else:
                    self.logger.warning(f'No config.yaml in {dir_path}')

    def run_experiment(self, dir_path):
        """
        Executes an experiment based on the 'config.yaml' file in the given directory.
        """

        with open(os.path.join(dir_path, "config.yaml"), "r") as config_file:
            self.config = yaml.safe_load(config_file)

        self.experiment_logger = setup_logging("EXPERIMENT", output_file=dir_path, config = self.config)

        #load any previous results (in "detailed_results.json") as dictionary
        results = self.load_results(dir_path)
        #results = {
        #    query_index: {
        #        'ranked_list': [<list of ranked docIDs>]
        #    },
        #    ...
        #  }

        #get dictionary of data paths:
        data_path_dict = self.setup_data_paths(self.config)
        #e.g. data_path_dict = 
            #{"embeddings_path": "emb.pkl", "text_path": "collection.jsonl"}


        

    def setup_data_paths(self, config):
        """
        Read config to see which keys need to be included in the data_path_dict. Empty keys will not be included.
        """
        data_paths_config = config.get('data_paths', {})
        data_path_dict = {key: path for key, path in data_paths_config.items() if isinstance(path, str) and path.strip()}
        return data_path_dict

    def load_past_results(self, dir_path):
        detailed_results_path = os.path.join(dir_path, 'detailed_results.json')
        if os.path.exists(detailed_results_path):
            with open(detailed_results_path, 'r') as file:
                return json.load(file)
        return {}