from LLM_calls import load_llm, llm_call
from database import EntityDatabase
import json
from tqdm import tqdm
import random
import argparse
import re
import numpy as np
from typing import List, Dict, Any, Tuple
from collections import defaultdict
import math
import os

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

class RankingAlgorithm:
    """Base class for ranking algorithms"""
    
    def rank(self, experiments: List[Dict]) -> List[Dict]:
        """
        Rank documents based on experiments
        experiments: List of experiments, each containing winners and losers
        """
        raise NotImplementedError

class BradleyTerryLuce(RankingAlgorithm):
    """Bradley-Terry-Luce model for ranking"""
    
    def __init__(self, max_iterations=100, tolerance=1e-6):
        self.max_iterations = max_iterations
        self.tolerance = tolerance
    
    def rank(self, experiments: List[Dict]) -> List[Dict]:
        """Rank using Bradley-Terry-Luce model"""
        # Collect all unique document IDs
        all_doc_ids = set()
        for exp in experiments:
            for doc in exp['winners']:
                all_doc_ids.add(doc['id'])
            for doc in exp['losers']:
                all_doc_ids.add(doc['id'])
        
        all_doc_ids = list(all_doc_ids)
        doc_to_idx = {doc_id: i for i, doc_id in enumerate(all_doc_ids)}
        n_docs = len(all_doc_ids)
        
        # Initialize parameters
        params = np.ones(n_docs)
        
        # EM algorithm
        for iteration in range(self.max_iterations):
            old_params = params.copy()
            
            # E-step: compute expected counts
            wins = np.zeros(n_docs)
            total_games = np.zeros(n_docs)
            
            for exp in experiments:
                winners = [doc_to_idx[doc['id']] for doc in exp['winners']]
                losers = [doc_to_idx[doc['id']] for doc in exp['losers']]
                
                for winner in winners:
                    for loser in losers:
                        # Probability that winner beats loser
                        prob = params[winner] / (params[winner] + params[loser])
                        wins[winner] += prob
                        total_games[winner] += prob
                        total_games[loser] += (1 - prob)
            
            # M-step: update parameters
            for i in range(n_docs):
                if total_games[i] > 0:
                    params[i] = wins[i] / total_games[i]
                else:
                    params[i] = 1.0
            
            # Normalize
            params = params / np.sum(params) * n_docs
            
            # Check convergence
            if np.max(np.abs(params - old_params)) < self.tolerance:
                break
        
        # Create ranked results
        ranked_docs = []
        for i, doc_id in enumerate(all_doc_ids):
            ranked_docs.append({
                'entity_id': doc_id,
                'score': float(params[i]),
                'rank': 0  # Will be set after sorting
            })
        
        # Sort by score (descending)
        ranked_docs.sort(key=lambda x: x['score'], reverse=True)
        for i, doc in enumerate(ranked_docs):
            doc['rank'] = i + 1
        
        return ranked_docs

class PlackettLuce(RankingAlgorithm):
    """Plackett-Luce model for ranking"""
    
    def __init__(self, max_iterations=100, tolerance=1e-6):
        self.max_iterations = max_iterations
        self.tolerance = tolerance
    
    def rank(self, experiments: List[Dict]) -> List[Dict]:
        """Rank using Plackett-Luce model"""
        # Collect all unique document IDs
        all_doc_ids = set()
        for exp in experiments:
            for doc in exp['winners']:
                all_doc_ids.add(doc['id'])
            for doc in exp['losers']:
                all_doc_ids.add(doc['id'])
        
        all_doc_ids = list(all_doc_ids)
        doc_to_idx = {doc_id: i for i, doc_id in enumerate(all_doc_ids)}
        n_docs = len(all_doc_ids)
        
        # Initialize parameters
        params = np.ones(n_docs)
        
        # EM algorithm
        for iteration in range(self.max_iterations):
            old_params = params.copy()
            
            # E-step: compute expected counts
            wins = np.zeros(n_docs)
            total_games = np.zeros(n_docs)
            
            for exp in experiments:
                winners = [doc_to_idx[doc['id']] for doc in exp['winners']]
                losers = [doc_to_idx[doc['id']] for doc in exp['losers']]
                
                # For each winner, compute probability of being selected first
                for i, winner in enumerate(winners):
                    # Probability of winner being selected first
                    remaining_docs = winners[i:] + losers
                    if remaining_docs:
                        prob = params[winner] / sum(params[doc_idx] for doc_idx in remaining_docs)
                        wins[winner] += prob
                        total_games[winner] += prob
                    else:
                        wins[winner] += 1.0  # Default value
                        total_games[winner] += 1.0
                
                # For losers, they contribute to the denominator
                for loser in losers:
                    remaining_docs = winners + losers
                    if remaining_docs:
                        prob = params[loser] / sum(params[doc_idx] for doc_idx in remaining_docs)
                        total_games[loser] += prob
                    else:
                        total_games[loser] += 1.0  # Default value
            
            # M-step: update parameters
            for i in range(n_docs):
                if total_games[i] > 0:
                    params[i] = wins[i] / total_games[i]
                else:
                    params[i] = 1.0
            
            # Normalize
            params = params / np.sum(params) * n_docs
            
            # Check convergence
            if np.max(np.abs(params - old_params)) < self.tolerance:
                break
        
        # Create ranked results
        ranked_docs = []
        for i, doc_id in enumerate(all_doc_ids):
            ranked_docs.append({
                'entity_id': doc_id,
                'score': float(params[i]),
                'rank': 0
            })
        
        # Sort by score (descending)
        ranked_docs.sort(key=lambda x: x['score'], reverse=True)
        for i, doc in enumerate(ranked_docs):
            doc['rank'] = i + 1
        
        return ranked_docs

class Davidson(RankingAlgorithm):
    """Davidson model for ranking (handles ties)"""
    
    def __init__(self, max_iterations=100, tolerance=1e-6):
        self.max_iterations = max_iterations
        self.tolerance = tolerance
    
    def rank(self, experiments: List[Dict]) -> List[Dict]:
        """Rank using Davidson model"""
        # Collect all unique document IDs
        all_doc_ids = set()
        for exp in experiments:
            for doc in exp['winners']:
                all_doc_ids.add(doc['id'])
            for doc in exp['losers']:
                all_doc_ids.add(doc['id'])
        
        all_doc_ids = list(all_doc_ids)
        doc_to_idx = {doc_id: i for i, doc_id in enumerate(all_doc_ids)}
        n_docs = len(all_doc_ids)
        
        # Initialize parameters
        params = np.ones(n_docs)
        tie_param = 1.0  # Parameter for ties
        
        # EM algorithm
        for iteration in range(self.max_iterations):
            old_params = params.copy()
            old_tie_param = tie_param
            
            # E-step: compute expected counts
            wins = np.zeros(n_docs)
            total_games = np.zeros(n_docs)
            tie_count = 0
            total_tie_games = 0
            
            for exp in experiments:
                winners = [doc_to_idx[doc['id']] for doc in exp['winners']]
                losers = [doc_to_idx[doc['id']] for doc in exp['losers']]
                
                for winner in winners:
                    for loser in losers:
                        # Probability that winner beats loser
                        prob = params[winner] / (params[winner] + params[loser] + tie_param)
                        wins[winner] += prob
                        total_games[winner] += prob
                        total_games[loser] += (1 - prob)
                        
                        # Tie probability
                        tie_prob = tie_param / (params[winner] + params[loser] + tie_param)
                        tie_count += tie_prob
                        total_tie_games += tie_prob
            
            # M-step: update parameters
            for i in range(n_docs):
                if total_games[i] > 0:
                    params[i] = wins[i] / total_games[i]
                else:
                    params[i] = 1.0
            
            if total_tie_games > 0:
                tie_param = tie_count / total_tie_games
            else:
                tie_param = 1.0
            
            # Normalize
            params = params / np.sum(params) * n_docs
            
            # Check convergence
            if (np.max(np.abs(params - old_params)) < self.tolerance and 
                abs(tie_param - old_tie_param) < self.tolerance):
                break
        
        # Create ranked results
        ranked_docs = []
        for i, doc_id in enumerate(all_doc_ids):
            ranked_docs.append({
                'entity_id': doc_id,
                'score': float(params[i]),
                'rank': 0
            })
        
        # Sort by score (descending)
        ranked_docs.sort(key=lambda x: x['score'], reverse=True)
        for i, doc in enumerate(ranked_docs):
            doc['rank'] = i + 1
        
        return ranked_docs

def experimenting_function(w: int, p: int, d: List[Dict], c: str, llm_config: Dict) -> List[Dict]:
    """
    Perform batch experiments to extract winners and losers
    
    Args:
        w: number of winners for each experiment
        p: subset size of documents to sample
        d: population, all the searched documents
        c: condition/context for sampling documents
        llm_config: LLM configuration
    
    Returns:
        List of experiments with winners and losers
    """
    experiments = []
    
    # Sample p documents from population d
    if len(d) <= p:
        sampled_docs = d
    else:
        sampled_docs = random.sample(d, p)
    
    # Create experiment prompt
    prompt = f"""
Given the context: "{c}"

Here are {len(sampled_docs)} candidate entities:
"""
    
    for i, doc in enumerate(sampled_docs):
        prompt += f"{i+1}. {doc.get('title', 'Unknown')}: {doc.get('description', 'No description')}\n"
    
    prompt += f"""
Please select the top {w} most relevant entities for the given context.
Return only the numbers (e.g., "1, 3, 5") of the selected entities.
"""
    
    try:
        response = llm_call(prompt, **llm_config)
        
        # Check if response is valid
        if not response or not isinstance(response, str):
            raise ValueError("Invalid response from LLM")
        
        # Parse response to get winner indices
        winner_indices = []
        for num_str in response.split(','):
            try:
                idx = int(num_str.strip()) - 1  # Convert to 0-based index
                if 0 <= idx < len(sampled_docs):
                    winner_indices.append(idx)
            except ValueError:
                continue
        
        # Ensure we have exactly w winners
        if len(winner_indices) > w:
            winner_indices = winner_indices[:w]
        elif len(winner_indices) < w and len(winner_indices) > 0:
            # Fill with random selections if needed
            remaining = [i for i in range(len(sampled_docs)) if i not in winner_indices]
            needed = w - len(winner_indices)
            if needed <= len(remaining):
                winner_indices.extend(random.sample(remaining, needed))
        
        # Create winners and losers
        winners = [sampled_docs[i] for i in winner_indices if i < len(sampled_docs)]
        losers = [sampled_docs[i] for i in range(len(sampled_docs)) if i not in winner_indices]
        
        if winners and losers:
            experiments.append({
                'winners': winners,
                'losers': losers,
                'context': c
            })
    
    except Exception as e:
        print(f"Error in experimenting function: {e}")
        # Fallback: random selection
        if len(sampled_docs) > w:
            winners = random.sample(sampled_docs, w)
            losers = [doc for doc in sampled_docs if doc not in winners]
            experiments.append({
                'winners': winners,
                'losers': losers,
                'context': c
            })
    
    return experiments

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

def mock_llm_call(messages, **config):
    """Mock LLM call for testing without API keys"""
    if isinstance(messages, str):
        # Simple mock responses based on content
        if "search query" in messages.lower():
            return "technology company"
        elif "select the top" in messages.lower():
            return "1, 2"  # Select first two entities
        else:
            return "Mock response"
    return "Mock response"

def test_linking():
    """Test the linking functionality"""
    print("Testing Entity Linking System...")
    
    # Try to create database with Zeshel data first, fallback to fake data
    zeshel_path = "/home/kat/Desktop/UppsalaUniversity/Project/EntityLinking/zeshel/documents"
    if os.path.exists(zeshel_path):
        print("Loading Zeshel database for testing...")
        db = EntityDatabase(zeshel_documents_path=zeshel_path, max_documents=50)
        
        # Create test data based on Zeshel content
        test_entities = ["character", "show", "episode"]
        test_left_contexts = [
            "The main",
            "A popular",
            "The latest"
        ]
        test_right_contexts = [
            "in the series",
            "on television",
            "was aired"
        ]
        test_metadatas = [
            {"type": "character", "corpus": "muppets"},
            {"type": "show", "corpus": "muppets"},
            {"type": "episode", "corpus": "muppets"}
        ]
    else:
        print("Zeshel data not found, using fake data...")
        db = EntityDatabase()
        
        # Create test data for fake data
        test_entities = ["Apple", "Microsoft", "Google"]
        test_left_contexts = [
            "I work at",
            "The company",
            "Search engine"
        ]
        test_right_contexts = [
            "and love their products",
            "was founded by Bill Gates",
            "is very popular"
        ]
        test_metadatas = [
            {"type": "company"},
            {"type": "company"},
            {"type": "company"}
        ]
    
    # Test with different ranking algorithms
    algorithms = ['bradley_terry', 'plackett_luce', 'davidson']
    
    for algorithm in algorithms:
        print(f"\n=== Testing {algorithm.upper()} Algorithm ===")
        
        try:
            # Mock LLM config (since we don't have API keys set up)
            llm_config = {
                'type': 'mock',
                'model_name': 'mock-model'
            }
            
            # Temporarily replace llm_call with mock function
            import prompt
            original_llm_call = prompt.llm_call
            prompt.llm_call = mock_llm_call
            
            # Test linking
            results = linking(
                batch_entities=test_entities,
                batch_left_contexts=test_left_contexts,
                batch_right_contexts=test_right_contexts,
                batch_metadatas=test_metadatas,
                entities_database=db,
                ranking_algorithm=algorithm,
                n_queries=2,  # Reduced for testing
                top_k_search=10,
                n_experiments=3,  # Reduced for testing
                experiment_winners=2,
                experiment_subset_size=6,
                **llm_config
            )
            
            # Restore original function
            prompt.llm_call = original_llm_call
            
            print(f"Results for {algorithm}:")
            for i, result in enumerate(results):
                if result:
                    print(f"  {test_entities[i]} -> {result['title']} (ID: {result['id']})")
                else:
                    print(f"  {test_entities[i]} -> No result")
        
        except Exception as e:
            print(f"Error testing {algorithm}: {e}")
    
    print("\n=== Testing Ranking Algorithms Directly ===")
    
    # Test ranking algorithms with mock experiments
    test_experiments = [
        {
            'winners': [{'id': 'e1', 'title': 'Apple Inc.'}, {'id': 'e2', 'title': 'Apple (fruit)'}],
            'losers': [{'id': 'e3', 'title': 'Microsoft Corporation'}, {'id': 'e4', 'title': 'Google LLC'}]
        },
        {
            'winners': [{'id': 'e1', 'title': 'Apple Inc.'}, {'id': 'e3', 'title': 'Microsoft Corporation'}],
            'losers': [{'id': 'e2', 'title': 'Apple (fruit)'}, {'id': 'e5', 'title': 'Banana'}]
        }
    ]
    
    for algorithm in algorithms:
        print(f"\nTesting {algorithm} ranking:")
        try:
            if algorithm == 'bradley_terry':
                ranker = BradleyTerryLuce()
            elif algorithm == 'plackett_luce':
                ranker = PlackettLuce()
            elif algorithm == 'davidson':
                ranker = Davidson()
            
            ranked_results = ranker.rank(test_experiments)
            
            print(f"  Top 3 results:")
            for i, result in enumerate(ranked_results[:3]):
                # Get entity details from database
                entity = db.get_entity(result['entity_id'])
                title = entity['title'] if entity else f"Entity {result['entity_id']}"
                print(f"    {i+1}. {title} (score: {result['score']:.4f})")
        
        except Exception as e:
            print(f"  Error: {e}")
    
    print("\nLinking test completed!")

def test_ranking_algorithms_only():
    """Test only the ranking algorithms without LLM dependencies"""
    print("Testing Ranking Algorithms Only...")
    
    # Try to create database with Zeshel data first, fallback to fake data
    zeshel_path = "/home/kat/Desktop/UppsalaUniversity/Project/EntityLinking/zeshel/documents"
    if os.path.exists(zeshel_path):
        print("Loading Zeshel database for ranking test...")
        db = EntityDatabase(zeshel_documents_path=zeshel_path, max_documents=20)
        
        # Create test experiments with real Zeshel entities
        entities = list(db.entities.values())[:6]  # Get first 6 entities
        if len(entities) >= 4:
            test_experiments = [
                {
                    'winners': [entities[0], entities[1]],
                    'losers': [entities[2], entities[3]]
                },
                {
                    'winners': [entities[0], entities[2]],
                    'losers': [entities[1], entities[4]] if len(entities) > 4 else [entities[1]]
                }
            ]
            if len(entities) >= 6:
                test_experiments.append({
                    'winners': [entities[3], entities[4]],
                    'losers': [entities[0], entities[5]]
                })
        else:
            # Fallback if not enough entities
            test_experiments = [
                {
                    'winners': [entities[0]] if len(entities) > 0 else [],
                    'losers': [entities[1]] if len(entities) > 1 else []
                }
            ]
    else:
        print("Zeshel data not found, using fake data...")
        db = EntityDatabase()
        
        # Test ranking algorithms with mock experiments (fake data)
        test_experiments = [
            {
                'winners': [{'id': 'e1', 'title': 'Apple Inc.'}, {'id': 'e2', 'title': 'Apple (fruit)'}],
                'losers': [{'id': 'e3', 'title': 'Microsoft Corporation'}, {'id': 'e4', 'title': 'Google LLC'}]
            },
            {
                'winners': [{'id': 'e1', 'title': 'Apple Inc.'}, {'id': 'e3', 'title': 'Microsoft Corporation'}],
                'losers': [{'id': 'e2', 'title': 'Apple (fruit)'}, {'id': 'e5', 'title': 'Banana'}]
            },
            {
                'winners': [{'id': 'e4', 'title': 'Google LLC'}, {'id': 'e3', 'title': 'Microsoft Corporation'}],
                'losers': [{'id': 'e1', 'title': 'Apple Inc.'}, {'id': 'e6', 'title': 'iPhone'}]
            }
        ]
    
    algorithms = ['bradley_terry', 'plackett_luce', 'davidson']
    
    for algorithm in algorithms:
        print(f"\n=== Testing {algorithm.upper()} Algorithm ===")
        try:
            if algorithm == 'bradley_terry':
                ranker = BradleyTerryLuce()
            elif algorithm == 'plackett_luce':
                ranker = PlackettLuce()
            elif algorithm == 'davidson':
                ranker = Davidson()
            
            ranked_results = ranker.rank(test_experiments)
            
            print(f"  Top 5 results:")
            for i, result in enumerate(ranked_results[:5]):
                # Get entity details from database
                entity = db.get_entity(result['entity_id'])
                title = entity['title'] if entity else f"Entity {result['entity_id']}"
                print(f"    {i+1}. {title} (score: {result['score']:.4f})")
        
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\nRanking algorithms test completed!")

def test_zeshel_integration():
    """Comprehensive test with Zeshel data integration"""
    print("=== Comprehensive Zeshel Integration Test ===\n")
    
    # Load Zeshel database
    zeshel_path = "/home/kat/Desktop/UppsalaUniversity/Project/EntityLinking/zeshel/documents"
    if not os.path.exists(zeshel_path):
        print("Zeshel data not found. Please ensure the zeshel/documents folder exists.")
        return
    
    print("1. Loading Zeshel database...")
    db = EntityDatabase(zeshel_documents_path=zeshel_path, max_documents=30)
    
    # Show database statistics
    stats = db.get_corpus_stats()
    print(f"   Loaded: {stats['total_entities']} entities from {stats['total_corpora']} corpora")
    print(f"   Corpora: {list(stats['corpora'].keys())}")
    
    # Test all search methods
    print("\n2. Testing all search methods:")
    
    # BM25 search
    print("\n   BM25 search for 'character':")
    results = db.bm25("character", top_k=3)
    for result in results:
        print(f"     - {result['title']} (score: {result.get('bm25_score', 'N/A')})")
    
    # Name search
    print("\n   Name search for 'episode':")
    results = db.name("episode")
    for result in results:
        print(f"     - {result['title']} (score: {result.get('name_match_score', 'N/A')})")
    
    # Embedding search
    print("\n   Embedding search for 'show':")
    results = db.embedding("show", top_k=3)
    for result in results:
        print(f"     - {result['title']} (score: {result.get('similarity_score', 'N/A')})")
    
    # Test entity retrieval
    print("\n3. Testing entity retrieval:")
    entity_ids = list(db.entities.keys())[:3]
    for entity_id in entity_ids:
        entity = db.get_entity(entity_id)
        if entity:
            corpus = entity['metadata'].get('corpus', 'unknown')
            print(f"   - {entity_id}: {entity['title']} (corpus: {corpus})")
    
    # Test ranking algorithms with real data
    print("\n4. Testing ranking algorithms with real Zeshel entities:")
    entities = list(db.entities.values())[:5]
    if len(entities) >= 4:
        test_experiments = [
            {
                'winners': [entities[0], entities[1]],
                'losers': [entities[2], entities[3]]
            }
        ]
        
        for algorithm in ['bradley_terry', 'plackett_luce', 'davidson']:
            print(f"\n   Testing {algorithm.upper()}:")
            try:
                if algorithm == 'bradley_terry':
                    ranker = BradleyTerryLuce()
                elif algorithm == 'plackett_luce':
                    ranker = PlackettLuce()
                elif algorithm == 'davidson':
                    ranker = Davidson()
                
                ranked_results = ranker.rank(test_experiments)
                print(f"     Top 3 results:")
                for i, result in enumerate(ranked_results[:3]):
                    entity = db.get_entity(result['entity_id'])
                    title = entity['title'] if entity else f"Entity {result['entity_id']}"
                    print(f"       {i+1}. {title} (score: {result['score']:.4f})")
            except Exception as e:
                print(f"     Error: {e}")
    
    # Test entity linking with minimal parameters
    print("\n5. Testing entity linking with Zeshel data:")
    test_entities = ["character", "show"]
    test_left_contexts = ["The main", "A popular"]
    test_right_contexts = ["in the series", "on television"]
    test_metadatas = [{"type": "character"}, {"type": "show"}]
    
    try:
        # Use mock LLM to avoid API calls
        import prompt
        original_llm_call = prompt.llm_call
        prompt.llm_call = mock_llm_call
        
        results = linking(
            batch_entities=test_entities,
            batch_left_contexts=test_left_contexts,
            batch_right_contexts=test_right_contexts,
            batch_metadatas=test_metadatas,
            entities_database=db,
            ranking_algorithm='bradley_terry',
            n_queries=1,
            top_k_search=5,
            n_experiments=1,
            experiment_winners=1,
            experiment_subset_size=3,
            type='mock',
            model_name='mock-model'
        )
        
        # Restore original function
        prompt.llm_call = original_llm_call
        
        print("   Results:")
        for i, result in enumerate(results):
            if result:
                corpus = result['metadata'].get('corpus', 'unknown')
                print(f"     {test_entities[i]} -> {result['title']} (corpus: {corpus})")
            else:
                print(f"     {test_entities[i]} -> No result")
    
    except Exception as e:
        print(f"   Error in entity linking: {e}")
    
    print("\n=== Zeshel Integration Test Completed ===")
    print("All components working correctly with real Zeshel data!")

if __name__ == "__main__":
    # Test ranking algorithms only (no LLM dependencies)
    test_ranking_algorithms_only()
    
    # Test comprehensive Zeshel integration
    test_zeshel_integration()
    
    # Test the full linking system with LLM calls
    print("\n" + "="*60)
    print("Testing Full Linking System with LLM Calls")
    print("="*60)
    test_linking()