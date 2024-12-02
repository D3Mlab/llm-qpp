import copy
import json
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
        reformed_q = self.llm.prompt(prompt)["message"]

        state["queries"].append(reformed_q)
        return state

    def rerank_best_and_latest(self, state):
        #given a) the previously best list of length K and b) K additional retrieved docs
        #rerank these 2xK docs to a list of length K

        curr_top_k_docIDs = state.get("curr_top_k_docIDs", [])

        # If curr_top_k_doc_ids is empty, use the top_k from last_k_retrieved_doc_ids
        if not curr_top_k_docIDs:
            curr_top_k_docIDs = copy.deepcopy(state.get("last_k_retrieved_docIDs"))

        corpus_path = self.config['data_paths']['corpus_text_path']
        doc_ids_and_texts = get_doc_text_list(curr_top_k_docIDs,corpus_path)
        #format: [{"docID": d1, "text": <text_d1>},...]


        prompt_dict = {
            'query' : self.get_init_q(state),
            'doc_ids_and_texts' : doc_ids_and_texts,
            'k' : len(doc_ids_and_texts)
            }

        template_dir = self.template_config["listwise_reranking"]
        prompt = self.render_prompt(prompt_dict, template_dir)
        self.logger.debug(f"reranking prompt: {prompt}")
        llm_output = self.llm.prompt(prompt)["message"]
        
        try:
            #get's list of docIDs as a string and converts to a list
            #e.g. "["d1","d3"]" -> ["d1","d3"]
            #todo - switch to regex parsing if outputs aren't consistent but still include a ranked list
            curr_top_k_docIDs = json.loads(llm_output)
            if not isinstance(curr_top_k_docIDs, list) or not all(isinstance(item, str) for item in curr_top_k_docIDs):
                self.logger.warning(f"Unexpected LLM output format: {curr_top_k_docIDs}")
        except json.JSONDecodeError:
            self.logger.warning(f"Failed to parse LLM output as list of docIDs: {llm_output}")
            curr_top_k_docIDs = []

        self.logger.debug(f"LLM rerarnking output: {curr_top_k_docIDs}")
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
        self.logger.debug(f"termination prompt: {prompt}")
        llm_output = self.llm.prompt(prompt)["message"]
        
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
        self.logger.debug(f"post-retrieval reformulation prompt: {prompt}")
        llm_output = self.llm.prompt(prompt)["message"]
        
        state["queries"].append(llm_output)
        return state


    def get_init_q(self,state):
        return state["queries"][0]

    def render_prompt(self, prompt_dict, template_dir):
        template = self.jinja_env.get_template(template_dir)
        return template.render(prompt_dict)

