
import json
import pandas as pd


#read "corpus_filepath" ndjson of format {"doc_id1" : {"text": <docID1 TEXT>}, ...} one doc at a time
#compute doc embedding via "embedder_name" class
#append docID and embedding to pickle file by pickling {"doc_id": doc_id, "embedding": tensor}
