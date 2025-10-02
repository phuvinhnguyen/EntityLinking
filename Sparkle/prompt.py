from LLM_calls import load_llm, llm_call
import json
from tqdm import tqdm
import random
import argparse
import re

def ner(sentences, **config):
    prompt = "Wrap all named entities in {{}}.\nSentence: {sent}\nOutput:"
    outputs = [llm_call(prompt.format(sent=s), **config) for s in sentences]
    batch_entities = []
    for out in outputs:
        entities = []
        starts = []
        for i, char in enumerate(out):
            if char == '{': starts.append(i)
            elif char == '}': entities.append(out[starts.pop():i].replace('{', '').replace('}', ''))
        batch_entities.append(entities)
    return batch_entities

def linking(
    batch_entities,
    batch_left_contexts,
    batch_right_contexts,
    batch_metadatas,
    entities_database
):
    '''
    batch_entities: batch (list) of entities
    batch_left_contexts: batch (list) of left contexts
    batch_right_contexts: batch (list) of right contexts
    batch_metadatas: batch (list) of metadatas
    entities_database: database of entities

    output: batch (list) of list of possible linked entities

    entities_database methods:
    - embedding(query) -> list of entities
    - name(entity name) -> list of entities
    - bm25(query) -> list of entities
    - related(entity id) -> list of entities
    - add_relation(entity_id1, relation_id, entity_id2)
    '''

    # Flow:
    # 1. get the top k entities from the database
    # 2. 
    pass