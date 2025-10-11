#!/usr/bin/env python3
"""
Evaluation script for Entity Linking system using proper evaluation setup
"""
import os
import sys
import json
import time
import argparse
from typing import List, Dict, Any, Tuple, Set
from collections import defaultdict
import numpy as np
from tqdm import tqdm

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from systems.system_factory import SystemFactory

class NERMetrics:
    """Metrics for Named Entity Recognition"""
    
    def __init__(self):
        self.total_tp = 0
        self.total_fp = 0
        self.total_fn = 0
        self.sample_count = 0
        self.type_metrics = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
    
    def update(self, eval_result: Dict[str, Any], entity_types: Dict[str, str] = None):
        """Update metrics with new evaluation result"""
        self.total_tp += eval_result['tp']
        self.total_fp += eval_result['fp']
        self.total_fn += eval_result['fn']
        self.sample_count += 1
        
        # Update per-type metrics if available
        if entity_types:
            for entity_type, counts in entity_types.items():
                self.type_metrics[entity_type]['tp'] += counts.get('tp', 0)
                self.type_metrics[entity_type]['fp'] += counts.get('fp', 0)
                self.type_metrics[entity_type]['fn'] += counts.get('fn', 0)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary metrics"""
        if self.sample_count == 0:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'type_metrics': {}}
        
        precision = self.total_tp / (self.total_tp + self.total_fp) if (self.total_tp + self.total_fp) > 0 else 0.0
        recall = self.total_tp / (self.total_tp + self.total_fn) if (self.total_tp + self.total_fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Calculate per-type metrics
        type_metrics = {}
        for entity_type, counts in self.type_metrics.items():
            tp, fp, fn = counts['tp'], counts['fp'], counts['fn']
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            type_metrics[entity_type] = {'precision': p, 'recall': r, 'f1': f, 'tp': tp, 'fp': fp, 'fn': fn}
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'total_tp': self.total_tp,
            'total_fp': self.total_fp,
            'total_fn': self.total_fn,
            'sample_count': self.sample_count,
            'type_metrics': type_metrics
        }

class LinkingMetrics:
    """Metrics for Entity Linking"""
    
    def __init__(self):
        self.total_tp = 0
        self.total_fp = 0
        self.total_fn = 0
        self.sample_count = 0
        self.type_metrics = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
    
    def update(self, eval_result: Dict[str, Any], entity_types: Dict[str, str] = None):
        """Update metrics with new evaluation result"""
        self.total_tp += eval_result['tp']
        self.total_fp += eval_result['fp']
        self.total_fn += eval_result['fn']
        self.sample_count += 1
        
        # Update per-type metrics if available
        if entity_types:
            for entity_type, counts in entity_types.items():
                self.type_metrics[entity_type]['tp'] += counts.get('tp', 0)
                self.type_metrics[entity_type]['fp'] += counts.get('fp', 0)
                self.type_metrics[entity_type]['fn'] += counts.get('fn', 0)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary metrics"""
        if self.sample_count == 0:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'type_metrics': {}}
        
        precision = self.total_tp / (self.total_tp + self.total_fp) if (self.total_tp + self.total_fp) > 0 else 0.0
        recall = self.total_tp / (self.total_tp + self.total_fn) if (self.total_tp + self.total_fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Calculate per-type metrics
        type_metrics = {}
        for entity_type, counts in self.type_metrics.items():
            tp, fp, fn = counts['tp'], counts['fp'], counts['fn']
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            type_metrics[entity_type] = {'precision': p, 'recall': r, 'f1': f, 'tp': tp, 'fp': fp, 'fn': fn}
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'total_tp': self.total_tp,
            'total_fp': self.total_fp,
            'total_fn': self.total_fn,
            'sample_count': self.sample_count,
            'type_metrics': type_metrics
        }

class CandidateMetrics:
    """Metrics for Candidate Generation"""
    
    def __init__(self):
        self.recall_at_1_sum = 0.0
        self.recall_at_5_sum = 0.0
        self.recall_at_10_sum = 0.0
        self.avg_candidates_sum = 0.0
        self.sample_count = 0
        self.candidate_lengths = []
    
    def update(self, eval_result: Dict[str, Any]):
        """Update metrics with new evaluation result"""
        self.recall_at_1_sum += eval_result['recall_at_1']
        self.recall_at_5_sum += eval_result['recall_at_5']
        self.recall_at_10_sum += eval_result['recall_at_10']
        self.avg_candidates_sum += eval_result['avg_candidates']
        self.sample_count += 1
        self.candidate_lengths.append(eval_result['avg_candidates'])
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary metrics"""
        if self.sample_count == 0:
            return {'recall_at_1': 0.0, 'recall_at_5': 0.0, 'recall_at_10': 0.0, 'avg_candidates': 0.0}
        
        return {
            'recall_at_1': self.recall_at_1_sum / self.sample_count,
            'recall_at_5': self.recall_at_5_sum / self.sample_count,
            'recall_at_10': self.recall_at_10_sum / self.sample_count,
            'avg_candidates': self.avg_candidates_sum / self.sample_count,
            'min_candidates': min(self.candidate_lengths) if self.candidate_lengths else 0,
            'max_candidates': max(self.candidate_lengths) if self.candidate_lengths else 0,
            'sample_count': self.sample_count
        }

class EntityLinkingEvaluator:
    """Evaluator for entity linking systems"""
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.system = None
        
    def setup(self, entities_path: str, system_name: str = "simple"):
        """Setup the evaluator with a specific system"""
        print(f"Setting up Entity Linking Evaluator with {system_name} system...")
        
        try:
            # Create system using factory
            self.system = SystemFactory.create_system(system_name, self.config)
            
            # Initialize system with entity database
            if not self.system.initialize(entities_path):
                print(f"Failed to initialize {system_name} system")
                return False
            
            print(f"Evaluator setup complete with {system_name} system")
            return True
            
        except Exception as e:
            print(f"Error setting up {system_name} system: {e}")
            return False
    
    def evaluate_sample(self, text: str, ground_truth: List[Dict]) -> Dict[str, Any]:
        """Evaluate a single sample using the configured system"""
        if not self.system:
            raise ValueError("System not initialized. Call setup() first.")
        
        # Use system to link entities
        result = self.system.link_entities(text, timeout=60)
        
        # Convert system result to evaluation format
        linked_entities = []
        for entity in result.entities:
            linked_entities.append({
                'mention': entity.mention,
                'entity_id': entity.entity_id,
                'entity_title': entity.entity_title,
                'start_pos': entity.start_pos,
                'end_pos': entity.end_pos,
                'confidence': entity.confidence
            })
        
        # For NER evaluation, we need detected entities
        # This is a limitation - we'll use linked entities as proxy
        detected_entities = []
        for entity in result.entities:
            # Create a simple detected entity object
            class SimpleDetectedEntity:
                def __init__(self, mention, start_pos, end_pos, entity_type="UNKNOWN"):
                    self.text = mention
                    self.start_pos = start_pos
                    self.end_pos = end_pos
                    self.entity_type = entity_type
            
            detected_entities.append(SimpleDetectedEntity(
                entity.mention, entity.start_pos, entity.end_pos
            ))
        
        # Evaluate NER aspect
        ner_eval = self._evaluate_ner(detected_entities, ground_truth)
        
        # Evaluate linking aspect
        linking_eval = self._evaluate_linking(linked_entities, ground_truth)
        
        # Evaluate candidate generation
        candidate_eval = self._evaluate_candidates(linked_entities, ground_truth)
        
        return {
            'text': text,
            'detected_entities': len(detected_entities),
            'linked_entities': len(linked_entities),
            'ground_truth_entities': len(ground_truth),
            'linked_results': linked_entities,
            'ground_truth': ground_truth,
            'ner_evaluation': ner_eval,
            'linking_evaluation': linking_eval,
            'candidate_evaluation': candidate_eval
        }
    
    def _evaluate_linking(self, linked_entities: List[Dict], ground_truth: List[Dict]) -> Dict[str, float]:
        """Evaluate linking results against ground truth with flexible span matching"""
        # Convert to sets for comparison with flexible matching
        predicted = set()
        for entity in linked_entities:
            # Use entity text and entity_id for matching (more flexible than exact spans)
            predicted.add((entity['mention'].lower(), entity['entity_id']))
        
        true_entities = set()
        for label in ground_truth:
            name = label.get('name', '').lower()
            entity_id = label.get('entity_id', '')
            if name and entity_id:
                true_entities.add((name, entity_id))
        
        # Calculate metrics
        if not predicted and not true_entities:
            return {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'tp': 0, 'fp': 0, 'fn': 0}
        
        if not predicted:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'tp': 0, 'fp': 0, 'fn': len(true_entities)}
        
        if not true_entities:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'tp': 0, 'fp': len(predicted), 'fn': 0}
        
        tp = len(predicted & true_entities)
        fp = len(predicted - true_entities)
        fn = len(true_entities - predicted)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'fn': fn
        }
    
    def evaluate_dataset(self, test_file: str, comprehensive: bool = False) -> Dict[str, Any]:
        """Evaluate on entire dataset with optional comprehensive metrics"""
        print(f"Evaluating on dataset: {test_file}")
        
        # Load test data
        test_samples = []
        with open(test_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    sample = json.loads(line.strip())
                    test_samples.append(sample)
                except json.JSONDecodeError:
                    continue
        
        print(f"Loaded {len(test_samples)} test samples")
        
        if comprehensive:
            return self._evaluate_comprehensive(test_samples)
        else:
            return self._evaluate_basic(test_samples)
    
    def _evaluate_basic(self, test_samples: List[Dict]) -> Dict[str, Any]:
        """Basic evaluation (original method)"""
        # Evaluate each sample
        results = []
        total_start_time = time.time()
        
        for i, sample in enumerate(test_samples):
            print(f"Processing sample {i+1}/{len(test_samples)}")
            
            text = sample.get('text', '')
            ground_truth = sample.get('labels', [])
            
            if not text:
                continue
            
            result = self.evaluate_sample(text, ground_truth)
            result['sample_id'] = sample.get('id', i)
            results.append(result)
            
            # Print sample results
            eval_metrics = result['linking_evaluation']
            print(f"  Sample {i+1}: P={eval_metrics['precision']:.3f}, R={eval_metrics['recall']:.3f}, F1={eval_metrics['f1']:.3f}")
            
            # Show detailed results
            print(f"    Detected: {result['detected_entities']}, Linked: {result['linked_entities']}, Ground Truth: {result['ground_truth_entities']}")
            
            # Show linked entities
            for entity in result['linked_results']:
                print(f"      '{entity['mention']}' -> {entity['entity_title']} (ID: {entity['entity_id']})")
        
        total_time = time.time() - total_start_time
        
        # Calculate overall metrics
        overall_metrics = self._calculate_overall_metrics(results)
        
        return {
            'total_samples': len(results),
            'total_time': total_time,
            'overall_metrics': overall_metrics,
            'detailed_results': results
        }
    
    def _evaluate_comprehensive(self, test_samples: List[Dict]) -> Dict[str, Any]:
        """Comprehensive evaluation with NER and linking metrics"""
        if not self.system:
            raise ValueError("System not initialized. Call setup() first.")
        
        # Initialize metrics
        ner_metrics = NERMetrics()
        linking_metrics = LinkingMetrics()
        candidate_metrics = CandidateMetrics()
        
        # Process each sample
        results = []
        total_start_time = time.time()
        
        for i, sample in enumerate(tqdm(test_samples, desc="Processing samples")):
            text = sample.get('text', '')
            ground_truth = sample.get('labels', [])
            
            if not text:
                continue
            
            # Use system to link entities
            result = self.system.link_entities(text, timeout=60)
            
            # Convert system result to evaluation format
            linked_entities = []
            for entity in result.entities:
                linked_entities.append({
                    'mention': entity.mention,
                    'entity_id': entity.entity_id,
                    'entity_title': entity.entity_title,
                    'start_pos': entity.start_pos,
                    'end_pos': entity.end_pos,
                    'confidence': entity.confidence,
                    'candidates': []  # Not available in current system interface
                })
            
            # For NER evaluation, we need detected entities
            # This is a limitation - we'll use linked entities as proxy
            detected_entities = []
            for entity in result.entities:
                # Create a simple detected entity object
                class SimpleDetectedEntity:
                    def __init__(self, mention, start_pos, end_pos, entity_type="UNKNOWN"):
                        self.text = mention
                        self.start_pos = start_pos
                        self.end_pos = end_pos
                        self.entity_type = entity_type
                
                detected_entities.append(SimpleDetectedEntity(
                    entity.mention, entity.start_pos, entity.end_pos
                ))
            
            # Evaluate NER aspect
            ner_eval = self._evaluate_ner(detected_entities, ground_truth)
            ner_metrics.update(ner_eval)
            
            # Evaluate linking aspect
            linking_eval = self._evaluate_linking(linked_entities, ground_truth)
            linking_metrics.update(linking_eval)
            
            # Evaluate candidate generation
            candidate_eval = self._evaluate_candidates(linked_entities, ground_truth)
            candidate_metrics.update(candidate_eval)
            
            result_dict = {
                'sample_id': sample.get('id', i),
                'text': text,
                'detected_entities': len(detected_entities),
                'linked_entities': len(linked_entities),
                'ground_truth_entities': len(ground_truth),
                'ner_evaluation': ner_eval,
                'linking_evaluation': linking_eval,
                'candidate_evaluation': candidate_eval,
                'linked_results': linked_entities,
                'ground_truth': ground_truth
            }
            results.append(result_dict)
        
        total_time = time.time() - total_start_time
        
        # Calculate overall metrics
        overall_metrics = {
            'total_samples': len(results),
            'total_time': total_time,
            'avg_time_per_sample': total_time / len(results) if results else 0,
            'ner_metrics': ner_metrics.get_summary(),
            'linking_metrics': linking_metrics.get_summary(),
            'candidate_metrics': candidate_metrics.get_summary()
        }
        
        return {
            'total_samples': len(results),
            'total_time': total_time,
            'overall_metrics': overall_metrics,
            'detailed_results': results
        }
    
    def _evaluate_ner(self, detected_entities: List, ground_truth: List[Dict]) -> Dict[str, Any]:
        """Evaluate NER aspect (entity detection)"""
        # Convert to sets for comparison
        predicted_spans = set()
        for entity in detected_entities:
            predicted_spans.add((entity.start_pos, entity.end_pos, entity.entity_type))
        
        true_spans = set()
        for label in ground_truth:
            span = label.get('span', [])
            if len(span) >= 2:
                true_spans.add((span[0], span[1], label.get('type', 'UNKNOWN')))
        
        # Calculate metrics
        if not predicted_spans and not true_spans:
            return {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'tp': 0, 'fp': 0, 'fn': 0}
        
        if not predicted_spans:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'tp': 0, 'fp': 0, 'fn': len(true_spans)}
        
        if not true_spans:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'tp': 0, 'fp': len(predicted_spans), 'fn': 0}
        
        tp = len(predicted_spans & true_spans)
        fp = len(predicted_spans - true_spans)
        fn = len(true_spans - predicted_spans)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'fn': fn
        }
    
    def _evaluate_candidates(self, linked_entities: List[Dict], ground_truth: List[Dict]) -> Dict[str, Any]:
        """Evaluate candidate generation aspect"""
        if not linked_entities:
            return {'recall_at_1': 0.0, 'recall_at_5': 0.0, 'recall_at_10': 0.0, 'avg_candidates': 0.0}
        
        # Create ground truth entity IDs
        true_entity_ids = set()
        for label in ground_truth:
            entity_id = label.get('entity_id', '')
            if entity_id:
                true_entity_ids.add(entity_id)
        
        recall_at_1 = 0
        recall_at_5 = 0
        recall_at_10 = 0
        total_candidates = 0
        valid_links = 0
        
        for entity in linked_entities:
            candidates = entity.get('candidates', [])
            total_candidates += len(candidates)
            
            if candidates:
                valid_links += 1
                
                # Check recall at different ranks
                candidate_ids = [c['id'] for c in candidates]
                
                # Find if any ground truth entity is in candidates
                found_at_1 = any(candidate_ids[0] == true_id for true_id in true_entity_ids) if candidate_ids else False
                found_at_5 = any(candidate_id in true_entity_ids for candidate_id in candidate_ids[:5]) if len(candidate_ids) >= 5 else False
                found_at_10 = any(candidate_id in true_entity_ids for candidate_id in candidate_ids[:10]) if len(candidate_ids) >= 10 else False
                
                if found_at_1:
                    recall_at_1 += 1
                if found_at_5:
                    recall_at_5 += 1
                if found_at_10:
                    recall_at_10 += 1
        
        return {
            'recall_at_1': recall_at_1 / valid_links if valid_links > 0 else 0.0,
            'recall_at_5': recall_at_5 / valid_links if valid_links > 0 else 0.0,
            'recall_at_10': recall_at_10 / valid_links if valid_links > 0 else 0.0,
            'avg_candidates': total_candidates / valid_links if valid_links > 0 else 0.0
        }
    
    def _calculate_overall_metrics(self, results: List[Dict]) -> Dict[str, float]:
        """Calculate overall evaluation metrics"""
        if not results:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        # Micro-averaged metrics (sum all TP, FP, FN)
        total_tp = sum(r['linking_evaluation']['tp'] for r in results)
        total_fp = sum(r['linking_evaluation']['fp'] for r in results)
        total_fn = sum(r['linking_evaluation']['fn'] for r in results)
        
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Macro-averaged metrics (average across samples)
        macro_precision = sum(r['linking_evaluation']['precision'] for r in results) / len(results)
        macro_recall = sum(r['linking_evaluation']['recall'] for r in results) / len(results)
        macro_f1 = sum(r['linking_evaluation']['f1'] for r in results) / len(results)
        
        return {
            'micro_precision': precision,
            'micro_recall': recall,
            'micro_f1': f1,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1,
            'total_tp': total_tp,
            'total_fp': total_fp,
            'total_fn': total_fn
        }
    
    def save_results(self, results: Dict[str, Any], output_file: str):
        """Save evaluation results to file"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {output_file}")

def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description='Entity Linking Evaluation')
    parser.add_argument('--entities_path', '-e', type=str, 
                       default='evaluation/entities/WikiEntities',
                       help='Path to entities (directory for Wikidata format or file for legacy format)')
    parser.add_argument('--test_file', '-t', type=str,
                       default='evaluation/test_data/test_cases.jsonl',
                       help='Path to test file')
    parser.add_argument('--output_file', '-o', type=str,
                       default='evaluation/results.json',
                       help='Path to output file')
    parser.add_argument('--comprehensive', '-c', action='store_true',
                       help='Run comprehensive evaluation with NER and linking metrics')
    parser.add_argument('--system', '-s', type=str, default='simple',
                       help='System to evaluate (simple, ranking)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    print("Entity Linking System - Evaluation")
    print("=" * 50)
    
    # Setup paths - try Wikidata format first, then fallback
    entities_path = args.entities_path
    if not os.path.exists(entities_path):
        # Try fallback
        fallback_path = "evaluation/entities/entities.json"
        if os.path.exists(fallback_path):
            entities_path = fallback_path
        else:
            print(f"Entities path not found: {args.entities_path}")
            return
    
    test_file = args.test_file
    output_file = args.output_file
    
    # Check if files exist
    if not os.path.exists(test_file):
        print(f"Test file not found: {test_file}")
        return
    
    # Create evaluator
    config = Config()
    config.TOP_K_SEARCH = 10
    config.N_EXPERIMENTS = 2
    config.EXPERIMENT_WINNERS = 2
    config.EXPERIMENT_SUBSET_SIZE = 4
    
    evaluator = EntityLinkingEvaluator(config)
    
    # Setup evaluator with specified system
    if not evaluator.setup(entities_path, args.system):
        print(f"Failed to setup evaluator with {args.system} system")
        return
    
    # Run evaluation
    results = evaluator.evaluate_dataset(test_file, comprehensive=args.comprehensive)
    
    # Print results
    print(f"\n{'='*50}")
    print("EVALUATION RESULTS")
    print(f"{'='*50}")
    
    overall = results['overall_metrics']
    print(f"Total samples: {results['total_samples']}")
    print(f"Total time: {results['total_time']:.2f}s")
    
    if args.comprehensive:
        print(f"Average time per sample: {overall['avg_time_per_sample']:.2f}s")
        
        # NER Metrics
        print(f"\n{'='*20} NER METRICS {'='*20}")
        ner_metrics = overall['ner_metrics']
        print(f"Precision: {ner_metrics['precision']:.4f}")
        print(f"Recall: {ner_metrics['recall']:.4f}")
        print(f"F1: {ner_metrics['f1']:.4f}")
        print(f"True Positives: {ner_metrics['total_tp']}")
        print(f"False Positives: {ner_metrics['total_fp']}")
        print(f"False Negatives: {ner_metrics['total_fn']}")
        
        # Linking Metrics
        print(f"\n{'='*20} LINKING METRICS {'='*20}")
        linking_metrics = overall['linking_metrics']
        print(f"Precision: {linking_metrics['precision']:.4f}")
        print(f"Recall: {linking_metrics['recall']:.4f}")
        print(f"F1: {linking_metrics['f1']:.4f}")
        print(f"True Positives: {linking_metrics['total_tp']}")
        print(f"False Positives: {linking_metrics['total_fp']}")
        print(f"False Negatives: {linking_metrics['total_fn']}")
        
        # Candidate Metrics
        print(f"\n{'='*20} CANDIDATE METRICS {'='*20}")
        candidate_metrics = overall['candidate_metrics']
        print(f"Recall@1: {candidate_metrics['recall_at_1']:.4f}")
        print(f"Recall@5: {candidate_metrics['recall_at_5']:.4f}")
        print(f"Recall@10: {candidate_metrics['recall_at_10']:.4f}")
        print(f"Average Candidates: {candidate_metrics['avg_candidates']:.2f}")
        print(f"Min Candidates: {candidate_metrics['min_candidates']}")
        print(f"Max Candidates: {candidate_metrics['max_candidates']}")
    else:
        print(f"Average time per sample: {results['total_time']/results['total_samples']:.2f}s")
        
        print(f"\nMicro-averaged metrics:")
        print(f"  Precision: {overall['micro_precision']:.3f}")
        print(f"  Recall: {overall['micro_recall']:.3f}")
        print(f"  F1: {overall['micro_f1']:.3f}")
        
        print(f"\nMacro-averaged metrics:")
        print(f"  Precision: {overall['macro_precision']:.3f}")
        print(f"  Recall: {overall['macro_recall']:.3f}")
        print(f"  F1: {overall['macro_f1']:.3f}")
        
        print(f"\nConfusion matrix:")
        print(f"  True Positives: {overall['total_tp']}")
        print(f"  False Positives: {overall['total_fp']}")
        print(f"  False Negatives: {overall['total_fn']}")
    
    # Save results
    evaluator.save_results(results, output_file)
    
    print(f"\n{'='*50}")
    print("Evaluation completed successfully!")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
