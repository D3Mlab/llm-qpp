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




#

        # Track total time spent waiting for LLM or MNLI
        self.total_llm_time = 0.0
        self.total_entailment_time = 0.0

        self.ITEM_SELECTION_MAP = {
        'greedy': self.item_selection_greedy,
        'random': self.item_selection_random,
        'entropy_reduction': self.item_selection_entropy_reduction,
        'ucb': self.item_selection_ucb,
        'thompson': self.item_selection_thompson,
        'best_and_most_uncertain': self.item_selection_top_and_most_uncertain,
        }
        
        self.ASPECT_EXTRACTION_MAP = {
        'val': self.get_aspect_val,
        'key_val': self.get_aspect_key_val
         }
    '''
    Generate an aspect from the item description on which to query 
    '''
    def get_aspect_key_val(self, item_desc):
        template_file = self.config['query']['aspect_gen_template']
        prompt_template = self.jinja_env.get_template(template_file)

        context = {
            "item_desc": item_desc, # Item description for the given item
            "aspects": self.aspects
        }
        prompt = prompt_template.render(context)

        self.logger.debug(prompt)

        aspect_pair = self.llm.make_request(prompt, temperature=self.config['llm']['temperature'])

        aspect_list = aspect_pair.split(",")
        for i in range(len(aspect_list)):
            aspect_list[i] = aspect_list[i].strip()
        
        aspect_dict = {"aspect_key": aspect_list[0], "aspect_value": aspect_list[1]}

        return aspect_dict

    def get_aspect_val(self, item_desc):
        template_file = self.config['query']['aspect_gen_template']
        prompt_template = self.jinja_env.get_template(template_file)

        context = {
            "item_desc": item_desc, # Item description for the given item
            "aspects": self.aspects
        }
        prompt = prompt_template.render(context)

        self.logger.debug(prompt)

        aspect_val = self.llm.make_request(prompt, temperature=self.config['llm']['temperature']).strip()
        
        aspect_dict = {"aspect_value": aspect_val}

        return aspect_dict
    
    '''
    Selects one items and generates a prompt for the language model based on that item
    '''
    def get_one_item_query(self):
        # Run item selection to get the item to generate from
        item_selection_method = self.ITEM_SELECTION_MAP[self.config['query']['item_selection']]
        # If it's the first turn, always use random
        if (len(self.queried_items) == 0): 
            item_selection_method = self.item_selection_random
        self.logger.debug(f"Selected Item with {item_selection_method.__name__}")
        top_item_id_list = item_selection_method() # should be single item list
        top_item_id = top_item_id_list[0]
        self.queried_items.append(top_item_id)
        item_desc = self.items[top_item_id]['description'] 
        self.logger.debug(f"itemId: {top_item_id} \n item description: {item_desc}")
        
        start = timeit.default_timer()


        # Get the aspect
        aspect_extraction_method = self.ASPECT_EXTRACTION_MAP[self.config['query']['aspect_extraction']]

        aspect_dict = aspect_extraction_method(item_desc)
        
        self.aspects.append(aspect_dict)

        # Generate query from aspect and item_desc
        template_file = self.config['query']['query_gen_template']
        prompt_template = self.jinja_env.get_template(template_file)

        context = {
            "item_desc": item_desc, # Item description for the given item
            "aspect_dict": aspect_dict
        }
        prompt = prompt_template.render(context)

        return prompt
    
    '''
    Selects two items and generates a prompt for the language model based on those items
    '''
    def get_two_item_query(self):
        # Run item selection to get the item to generate from
        item_selection_method = self.ITEM_SELECTION_MAP[self.config['query']['item_selection']]
        # If it's the first turn, always use random
        if (len(self.queried_items) == 0): 
            item_selection_method = self.item_selection_random
        self.logger.debug(f"Selected Item with {item_selection_method.__name__}")
        top_item_ids = item_selection_method(n=2)
        self.queried_items.append(top_item_ids)
        item_descs = [self.items[top_item_id]['description'] for top_item_id in top_item_ids]
        self.logger.debug(f"itemIds: {top_item_ids} \n item descriptions: {item_descs}")

        # Get the aspect
        aspect_extraction_method = self.ASPECT_EXTRACTION_MAP[self.config['query']['aspect_extraction']]

        aspect_dict = aspect_extraction_method(item_descs)
        
        self.aspects.append(aspect_dict)

        # Generate query from aspect and item_desc
        template_file = self.config['query']['query_gen_template']
        prompt_template = self.jinja_env.get_template(template_file)

        context = {
            "item_desc": item_descs, # Item description for the given item
            "aspect_dict": aspect_dict
        }
        prompt = prompt_template.render(context)

        return prompt


    '''
    Generates a query based on the current utility values and the provided set of items.
    '''
    def get_query(self):
        # NOTE: Hard coding this for now
        start = timeit.default_timer()
        #ANTON JUN 11: reverted to single query version
        #if (self.config['pe']['setup'] == "pairwise"):
        #    prompt = self.get_two_item_query()
        #else:
        prompt = self.get_one_item_query()

        self.logger.debug(prompt)

        user_query = self.llm.make_request(prompt, temperature=self.config['llm']['temperature'])
        stop = timeit.default_timer()
        # self.total_llm_time += (stop - start)
        self.logger.debug(user_query)
        return user_query
    
    '''
    Get the IDs of the top k recommended items
    '''
    def get_top_items(self, k=5):
        top_k_ids = heapq.nlargest(k, self.items, key=lambda i: (self.belief[i]['alpha'] / (self.belief[i]['alpha'] + self.belief[i]['beta']) ))
        return top_k_ids
    
    '''
    Update the model's beliefs, etc based on the user's response
    '''
    def update_from_response(self, query, response):
        interaction = {"query": query, "response": response}

        #update with latest aspect fields
        interaction.update(self.aspects[-1])
        
        self.interactions.append(interaction)

        #self.logger.debug(f"self.interactions: {self.interactions}")

        # Use either full history or just last response 
        preference = [self.interactions[-1]] if self.config['pe']['response_update']=="individual" else self.interactions

        #optional preprocessing
        if self.config['item_scoring']['preprocess_query']:
            preference = self.history_preprocessor.preprocess(preference)
            self.logger.debug("Preference: %s" % preference)

        #ANTON Dec 12 TODO: set truncation warnings
        #get like_prob for all items
        start = timeit.default_timer()
        like_probs = self.item_scorer.score_items(preference, self.items)
        stop = timeit.default_timer()
        self.total_entailment_time += (stop - start)

        for item_id in self.items:
            new_alpha = self.belief[item_id]['alpha'] + like_probs[item_id] # new_alpha = old_alpha + L
            new_beta = 1 - like_probs[item_id] + self.belief[item_id]['beta'] # new_beta = 1 - L + old_beta
            self.belief[item_id] = {'alpha': new_alpha, 'beta': new_beta}

            self.logger.debug("Like probs for item %s: %f, updated alpha = %f and beta = %f" % (item_id, like_probs[item_id], self.belief[item_id]['alpha'], self.belief[item_id]['beta']))

        # Deep copy belief state dict
        belief_copy = {}
        for key, belief in self.belief.items():
            belief_dict = {'alpha': belief['alpha'], 'beta': belief['beta']}
            belief_copy[key] = belief_dict
        self.all_beliefs.append(belief_copy)


        # Append the top k items to self.recs for return_dict
        k = self.config['pe']['num_recs']
        top_recs = self.get_top_items(k)
        self.recs.append(top_recs)


    def reset(self):
        # Random seed is now set in experiment_manager
        super().reset()
        self.belief = {}
        for id in self.items:
            self.belief[id] = {"alpha": 0.5, "beta": 0.5}
        self.queried_items = []
        self.all_beliefs = []
        self.aspects = []
        self.recs = [] # List of recommended items at each step. Number recommended is from config
        self.total_entailment_time = 0.0
        self.total_llm_time = 0.0


    '''
    the following item_selection_x() methods use different pointwise item selection methods. Each 
    method returns the item_id on which to query, based on the selection method.
    '''
    # Select the item_id with the highest expected utility.
    def item_selection_greedy(self, n=1):
        # Check for epsilon parameter
        if 'epsilon' in self.config['query']: # If no epsilon is provided, just do fully greedy
            eps = self.config['query']['epsilon']
            rand_draw = np.random.rand()
            if rand_draw < eps:
                top_id = np.random.choice(list(self.items))
                return top_id
        top_ids = heapq.nlargest(n, self.items, key=lambda i: (self.belief[i]['alpha'] / (self.belief[i]['alpha'] + self.belief[i]['beta'])))
        return top_ids # Return item ids as list

    # Select the item_id at random
    def item_selection_random(self, n=1):
        top_ids = np.random.choice(list(self.items), n).tolist()
        return top_ids

    # Select the item_id with the highest variance in utility
    def item_selection_entropy_reduction(self, n=1):
        top_id = heapq.nlargest(n, self.items, key=lambda i: (
            (self.belief[i]['alpha'] * self.belief[i]['beta'] * (self.belief[i]['alpha'] + self.belief[i]['beta'] + 1)) / 
            (math.pow(self.belief[i]['alpha'] + self.belief[i]['beta'], 2) * (self.belief[i]['alpha'] + self.belief[i]['beta'] + 1))
        ))
        return top_id