import json
import os

def get_doc_text_list(ids, corpus_path):
    #ids: list of doc_ids
    #return [{id: d1, text: <text_d1>},...]

    id_and_text_list = []

    for doc in jsonl_line_generator(corpus_path):
        if doc.get('docID') in ids:
            id_and_text_list.append({'docID': doc.get('docID'), 'text': doc.get('text')})

    return id_and_text_list

def jsonl_line_generator(path):
        with open(path, 'r', encoding='utf-8') as file:
            for line in file:
                yield json.loads(line)
