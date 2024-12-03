import json
import os
from utils.setup_logging import setup_logging

#class with some basic agent actions
class AgentLogic():

    def __init__(self, config):
        self.config = config
        self.logger = setup_logging(self.__class__.__name__, self.config)

    def check_max_q_reforms(self,state):
        max_q_reforms = self.config['agent'].get("max_q_reforms")
        num_q_reforms = len(state['queries'])-1 #number of total queries - 1 (for initial query)

        if num_q_reforms < max_q_reforms:
            state["terminate"] = False
        else:
            state["terminate"] = True

        return state

    def return_last_retrieved_list(self, state):
        #to be used when there is no reranking
        state["curr_top_k_docIDs"] = state['retrieved_lists'][-1]
        state["terminate"] = True
        return state

#other misc helper functions
def get_doc_text_list(ids, corpus_path):
    #ids: list of doc_ids
    #return [{"docID": d1, "text": <text_d1>},...]

    if not ids:
        return []

    id_set = set(ids)
    id_and_text_dict = {}

    # Iterate through the corpus and collect only the required documents
    for doc in jsonl_line_generator(corpus_path):
        doc_id = doc.get('docID')
        if doc_id in id_set:
            id_and_text_dict[doc_id] = doc.get('text')
            if len(id_and_text_dict) == len(ids):
                break

    id_and_text_list = [{'docID': doc_id, 'text': id_and_text_dict[doc_id]} for doc_id in ids if doc_id in id_and_text_dict]

    return id_and_text_list

def jsonl_line_generator(path):
        with open(path, 'r', encoding='utf-8') as file:
            for line in file:
                yield json.loads(line)
