from typing import List, Dict, Any
import random
import numpy as np
from LLM_calls import llm_call


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
