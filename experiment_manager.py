
import yaml
import json
import argparse
import os
import csv
from pathlib import Path
from utils.setup_logging import setup_logging
import search_agent
from pathlib import Path
import pickle


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

        #self.data_path_dict = self.setup_data_paths(self.config)
        #e.g. data_path_dict = 
            #{"embeddings_path": "emb.pkl", "text_path": "collection.jsonl", ...}

        agent = search_agent.AGENT_CLASSES[self.config['agent']['agent_class']]
        self.agent = agent(self.config)

        self.queries = self.get_queries()
        #e.g. = {q1: "q1 text", q2: "q2 text",...}

        self.results_dir = Path(config_dir) / 'per_query_results'
        self.results_dir.mkdir(exist_ok=True)
        
        #get set of already processed qIDs {q1,q3,...}
        existing_results = self.load_existing_results()
        
        self.rank_queries(self.queries, self.results_dir, existing_results)

    def rank_queries(self, queries, results_dir, existing_results):
        """
        Ranks the provided queries and saves results per query.
        """
        for qid, query in queries.items():
            if qid in existing_results:
                self.experiment_logger.info(f'Results already available for query: {qid} {query}')
                continue

            try:
                self.experiment_logger.info(f'Ranking query: {qid} {query}')
                result = self.agent.rank(query)
                #results = {
                #        'ranked_list': [<list of ranked docIDs>]
                #    ...
                #  }
                self.experiment_logger.info('Rank successful')
                
                # Create directory and write results only for successful queries
                self.write_query_result(results_dir, qid, result)
            except Exception as e:
                self.experiment_logger.error(f'Failed to rank query {qid}: {str(e)}')

    def write_query_result(self, results_dir, qid, result):
        """
        Writes the result for a single query to a JSON file and saves TREC results.
        If a query embedding is present, it is saved as a separate pickle file.
        """
        query_result_dir = results_dir / f"{qid}"
        query_result_dir.mkdir(exist_ok=False)
        detailed_results_path = query_result_dir / "detailed_results.json"
        trec_file_path = query_result_dir / "trec_results_raw.txt"
        query_embedding_path = query_result_dir / "query_embedding.pkl"

        # Extract and save query embedding if present

        #todo: update for multiple query embeddings

        query_embedding = result.pop('query_embedding', None)
        if query_embedding is not None:
            with open(query_embedding_path, 'wb') as embedding_file:
                pickle.dump(query_embedding, embedding_file)

        # Write detailed results to JSON file
        with open(detailed_results_path, 'w') as file:
            json.dump(result, file, default=str, indent=4)

        # Write TREC results
        trec_results = []
        ranked_list = result.get('ranked_list', [])
        for doc_index, doc_id in enumerate(ranked_list):
            score = len(ranked_list) - doc_index
            trec_results.append(f"{qid} Q0 {doc_id} {doc_index + 1} {score} my_run")

        with open(trec_file_path, "w") as trec_file:
            trec_file.write("\n".join(trec_results))

    def load_existing_results(self):
        """
        Loads existing results by checking for both 'detailed_results.json' and 
        'trec_results_raw.txt' files in the results directory.
        """
        existing_results = set()
        for root, dirs, _ in os.walk(self.results_dir):
           for directory in dirs:
               detailed_results_path = Path(root) / directory / "detailed_results.json"
               trec_results_path = Path(root) / directory / "trec_results_raw.txt"
               if detailed_results_path.exists() and trec_results_path.exists():
                    existing_results.add(directory)  # Directory name is assumed to be the qid
        return existing_results

    #def setup_data_path
    #s(self, config):
        """
        Read config to see which keys need to be included in the data_path_dict. Empty keys will not be included.
        """
    #    data_paths_config = config.get('data_paths', {})
    #    data_path_dict = {key: path for key, path in data_paths_config.items() if isinstance(path, str) and path.strip()}
    #    return data_path_dict

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
                return queries_dict
        except Exception as e:
            self.experiment_logger.error(f"Failed to read queries from {queries_path}: {e}")
            raise RuntimeError(f"Failed to read queries from {queries_path}: {e}")            