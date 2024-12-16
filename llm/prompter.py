import copy
import json
import re
import jinja2
from utils.setup_logging import setup_logging
from utils.utils import *
from . import LLM_CLASSES

class Prompter():
    
    def __init__(self,config):

        self.config = config 
        self.logger = setup_logging(self.__class__.__name__, self.config)

        self.llm_config = config.get('llm', {})
        self.model_class_name = self.llm_config.get('model_class')
        self.model_name = self.llm_config.get('model_name')
        self.template_dir = self.llm_config.get('template_dir')
        self.template_config = config.get("templates", {})

        model_class = LLM_CLASSES.get(self.model_class_name)
        self.llm = model_class(config,self.model_name)

        self.jinja_env = jinja2.Environment(loader=jinja2.FileSystemLoader(searchpath=self.template_dir))

    def reform_q_pre_retr(self, state):
        #reformulate initial query (pre-retrieval)

        init_q = self.get_init_q(state)

        prompt_dict = {"query" : init_q}
        template_dir = self.template_config["pre_retrieval_reformulation"]
        prompt = self.render_prompt(prompt_dict, template_dir)
        self.add_prompt_to_state(state,prompt)

        llm_output = self.llm.prompt(prompt)["message"]
        self.add_response_to_state(state,llm_output)
        reformed_q = llm_output

        state["queries"].append(reformed_q)
        return state

    def rerank_best_and_latest(self, state):
        #given a) the previously best list of length K and b) K additional retrieved docs
        #rerank these 2xK docs to a list of length K


        #could rewrite based on a slice to take of the retrieved docs (could set very high like 200)
        curr_top_k_docIDs = state.get("curr_top_k_docIDs", [])
        last_k_retrieved_docIDs = state["retrieved_lists"][-1]
        doc_ids = curr_top_k_docIDs + last_k_retrieved_docIDs
        k = self.config['rerank'].get('k')

        corpus_path = self.config['data_paths']['corpus_text_path']
        doc_ids_and_texts = get_doc_text_list(doc_ids,corpus_path)
        #format: [{"docID": d1, "text": <text_d1>},...]


        prompt_dict = {
            'query' : self.get_init_q(state),
            'doc_ids_and_texts' : doc_ids_and_texts,
            'k' : k
            }

        template_dir = self.template_config["reranking"]
        prompt = self.render_prompt(prompt_dict, template_dir)
        self.add_prompt_to_state(state,prompt)
        self.logger.debug(f"reranking prompt: {prompt}")

        llm_output = self.llm.prompt(prompt)["message"]
        self.add_response_to_state(state,llm_output)

        # Parse the LLM output
        curr_top_k_docIDs = self.parse_llm_list(llm_output)

        state["curr_top_k_docIDs"] = curr_top_k_docIDs
        return state

    def decide_termination_best_docs(self, state):
        #given the best K doc IDs (e.g. from reranking): curr_top_k_docIDs
        #decide whether to return results to user or continue search

        curr_top_k_docIDs = state.get("curr_top_k_docIDs", [])
        corpus_path = self.config['data_paths']['corpus_text_path']
        curr_top_k_ids_and_texts = get_doc_text_list(curr_top_k_docIDs,corpus_path)
        #format: [{"docID": d1, "text": <text_d1>},...]

        prompt_dict = {
            'query' : self.get_init_q(state),
            'curr_top_k_ids_and_texts' : curr_top_k_ids_and_texts
            }

        template_dir = self.template_config["termination"]
        prompt = self.render_prompt(prompt_dict, template_dir)
        self.add_prompt_to_state(state,prompt)
        #self.logger.debug(f"termination prompt: {prompt}")
        llm_output = self.llm.prompt(prompt)["message"]
        self.add_response_to_state(state,llm_output)
        
        # Process Yes/No termination decision from LLM output
        if isinstance(llm_output, str) and llm_output.strip().lower() == 'yes':
            state["terminate"] = True
        elif isinstance(llm_output, str) and llm_output.strip().lower() == 'no':
            state["terminate"] = False
        else:
            self.logger.warning(f"Unexpected LLM output for termination decision: {llm_output}")
            state["terminate"] = False  # Default to not terminate if response is unclear

        self.logger.debug(f"LLM termination decision output: {llm_output}")
        return state

    def reform_q_post_retr(self, state):
        #reformulate the query given various combinations of
        #-query reformulation history
        #-doc retrieval history
        #-current top K docs

        curr_top_k_docIDs = state.get("curr_top_k_docIDs", [])
        corpus_path = self.config['data_paths']['corpus_text_path']
        curr_top_k_ids_and_texts = get_doc_text_list(curr_top_k_docIDs,corpus_path)
        #format: [{"docID": d1, "text": <text_d1>},...]

        prompt_dict = {
            'init_q' : self.get_init_q(state),
            'reformed_qs' : state['queries'][1:],
            'curr_top_k_ids_and_texts' : curr_top_k_ids_and_texts
             #todo - add item - 'all_retrieved_doc_ids_and_texts' : # [[d^t=1_0,...d^t=0_K], ..., [d^t'=t_0,...d^t'=t_K]]
            }

        template_dir = self.template_config["post_retrieval_reformulation"]
        prompt = self.render_prompt(prompt_dict, template_dir)
        self.add_prompt_to_state(state,prompt)
        #self.logger.debug(f"post-retrieval reformulation prompt: {prompt}")
        llm_output = self.llm.prompt(prompt)["message"]
        self.add_response_to_state(state,llm_output)

        state["queries"].append(llm_output)
        return state

    def add_prompt_to_state(self,state,prompt):
        if "prompts" not in state:
            state["prompts"] = []  
        state["prompts"].append(prompt)

    def add_response_to_state(self,state,response):
        if "responses" not in state:
            state["responses"] = []  
        state["responses"].append(response)

    def get_init_q(self,state):
        return state["queries"][0]

    def render_prompt(self, prompt_dict, template_dir):
        template = self.jinja_env.get_template(template_dir)
        return template.render(prompt_dict)

    def parse_llm_list(self, llm_output):
        # Try parsing the LLM output using JSON list reader
        try:
            return json.loads(llm_output)
        except json.JSONDecodeError:
            # Use regex as a fallback to extract a list
            self.logger.warning(f"Could not parse LLM output as list of docIDs, using regex parsing to look for list in LLM output: {llm_output}")
            match = re.search(r'\[.*?\]', llm_output, re.DOTALL)
            if match:
                try:
                    # Extract and convert single-quoted strings to double quotes for JSON compatibility
                    extracted_list = match.group(0).replace("'", '"')
                    return json.loads(extracted_list)
                except json.JSONDecodeError:
                    self.logger.warning(f"Regex extraction failed to parse as JSON: {llm_output}")
    
        self.logger.warning(f"No valid list of docIDs found in LLM output: {llm_output}")
        return []