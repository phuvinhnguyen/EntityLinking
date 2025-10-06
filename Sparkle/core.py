from LLM_calls import load_llm, llm_call
from database import EntityDatabase
import json, os, random
from tqdm import tqdm
from algorithms import *
import numpy as np
from typing import List, Dict, Any, Tuple
from argparse import ArgumentParser


def ner(sentences, **config):
    '''
    Idea:
    - Sentence -> overlapsed chunks -> detect entities -> merge entities
    '''
    prompt = "Wrap all named entities in {{}}.\nSentence: {sent}\nOutput:"
    outputs = [llm_call(prompt.format(sent=s), **config) for s in sentences]
    batch_entities = []
    for sent, out in zip(sentences, outputs):
        entities = []
        starts = []
        for i, char in enumerate(out):
            if char == '{': starts.append(i)
            elif char == '}':
                start_idx = starts.pop()
                entities.append([out[start_idx:i].replace('{', '').replace('}', ''), out, start_idx])
        batch_entities.append(entities)
    return batch_entities

def linking(
    batch_entities,
    batch_left_contexts,
    batch_right_contexts,
    batch_metadatas,
    entities_database,
    ranking_algorithm='bradley_terry',
    n_queries=3,
    top_k_search=20,
    n_experiments=5,
    experiment_winners=2,
    experiment_subset_size=8,
    **llm_config
):
    '''
    batch_entities: batch (list) of entities
    batch_left_contexts: batch (list) of left contexts
    batch_right_contexts: batch (list) of right contexts
    batch_metadatas: batch (list) of metadatas
    entities_database: database of entities
    ranking_algorithm: 'bradley_terry', 'plackett_luce', or 'davidson'
    n_queries: number of different queries to generate per entity
    top_k_search: number of top entities to retrieve from database
    n_experiments: number of experiments to perform
    experiment_winners: number of winners per experiment
    experiment_subset_size: size of subset for each experiment

    output: batch (list) of best linked entities
    '''
    
    # Initialize ranking algorithm
    if ranking_algorithm == 'bradley_terry':
        ranker = BradleyTerryLuce()
    elif ranking_algorithm == 'plackett_luce':
        ranker = PlackettLuce()
    elif ranking_algorithm == 'davidson':
        ranker = Davidson()
    else:
        raise ValueError(f"Unknown ranking algorithm: {ranking_algorithm}")
    
    results = []
    
    for i, (entity, left_context, right_context, metadata) in enumerate(
        zip(batch_entities, batch_left_contexts, batch_right_contexts, batch_metadatas)
    ):
        print(f"Processing entity {i+1}/{len(batch_entities)}: {entity}")
        
        # Step 1: Generate n different queries for the entity
        queries = []
        for j in range(n_queries):
            query_prompt = f"""
Given the entity "{entity}" in the context:
Left: "{left_context}"
Right: "{right_context}"

Generate a search query to find relevant entities for this mention. 
Focus on different aspects: {['semantic meaning', 'contextual usage', 'domain-specific'][j % 3]}.

Query:"""
            
            try:
                query = llm_call(query_prompt, **llm_config)
                if query and isinstance(query, str):
                    queries.append(query.strip())
                else:
                    queries.append(entity)  # Fallback to entity name
            except Exception as e:
                print(f"Error generating query: {e}")
                queries.append(entity)  # Fallback to entity name
        
        # Step 2: Search database using all queries
        all_candidates = []
        for query in queries:
            # Try different search methods
            candidates = entities_database.embedding(query, top_k=top_k_search//n_queries)
            all_candidates.extend(candidates)
            
            # Also try name-based search
            name_candidates = entities_database.name(entity)
            all_candidates.extend(name_candidates[:3])  # Top 3 name matches
        
        # Remove duplicates and get top candidates
        seen_ids = set()
        unique_candidates = []
        for candidate in all_candidates:
            if candidate['id'] not in seen_ids:
                seen_ids.add(candidate['id'])
                unique_candidates.append(candidate)
        
        # Take top candidates
        top_candidates = unique_candidates[:top_k_search]
        
        if not top_candidates:
            results.append(None)
            continue
        
        # Step 3: Perform experiments
        context = f"Entity: {entity}, Left: {left_context}, Right: {right_context}"
        all_experiments = []
        
        for _ in range(n_experiments):
            experiments = experimenting_function(
                w=experiment_winners,
                p=min(experiment_subset_size, len(top_candidates)),
                d=top_candidates,
                c=context,
                llm_config=llm_config
            )
            all_experiments.extend(experiments)
        
        if not all_experiments:
            # Fallback: return top candidate by embedding score
            best_candidate = max(top_candidates, key=lambda x: x.get('similarity_score', 0))
            results.append(best_candidate)
            continue
        
        # Step 4: Rank using selected algorithm
        try:
            ranked_results = ranker.rank(all_experiments)
            
            if ranked_results:
                # Get the best ranked entity
                best_entity_id = ranked_results[0]['entity_id']
                best_candidate = entities_database.get_entity(best_entity_id)
                results.append(best_candidate)
            else:
                # Fallback
                best_candidate = max(top_candidates, key=lambda x: x.get('similarity_score', 0))
                results.append(best_candidate)
        
        except Exception as e:
            print(f"Error in ranking: {e}")
            # Fallback
            best_candidate = max(top_candidates, key=lambda x: x.get('similarity_score', 0))
            results.append(best_candidate)
    
    return results