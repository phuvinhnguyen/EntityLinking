"""
Base system interface for entity linking implementations
"""
import time
import json
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class EntityLink:
    """Represents a linked entity"""
    mention: str
    entity_id: str
    entity_title: str
    confidence: float
    start_pos: int
    end_pos: int
    context_left: str
    context_right: str
    metadata: Dict[str, Any] = None

@dataclass
class LinkingResult:
    """Result of entity linking process"""
    text: str
    entities: List[EntityLink]
    processing_time: float
    metadata: Dict[str, Any] = None

class BaseSystem(ABC):
    """Base class for all entity linking systems"""
    
    def __init__(self, config=None):
        self.config = config
        self.system_name = self.__class__.__name__
        self._initialized = False
    
    @abstractmethod
    def initialize(self, entities_path: str) -> bool:
        """
        Initialize the system with entity database
        
        Args:
            entities_path: Path to entity database (file or directory)
            
        Returns:
            bool: True if initialization successful, False otherwise
        """
        pass
    
    @abstractmethod
    def link_entities(self, text: str, timeout: int = 300) -> LinkingResult:
        """
        Link entities in the given text
        
        Args:
            text: Input text to process
            timeout: Maximum processing time in seconds
            
        Returns:
            LinkingResult: Results of entity linking
        """
        pass
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get information about the system"""
        return {
            'name': self.system_name,
            'initialized': self._initialized,
            'config': self.config.__dict__ if self.config else None
        }
    
    def process_text_file(self, input_file: str, output_file: str = None, timeout: int = 300) -> LinkingResult:
        """Process a text file and save results"""
        print(f"[{self.system_name}] Processing text file: {input_file}")
        
        # Read input text
        with open(input_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Link entities
        result = self.link_entities(text, timeout)
        
        # Save results
        if output_file is None:
            output_file = input_file.replace('.txt', f'_{self.system_name.lower()}_linked.json')
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'system_name': self.system_name,
                'text': result.text,
                'entities': [
                    {
                        'mention': e.mention,
                        'entity_id': e.entity_id,
                        'entity_title': e.entity_title,
                        'confidence': e.confidence,
                        'start_pos': e.start_pos,
                        'end_pos': e.end_pos,
                        'context_left': e.context_left,
                        'context_right': e.context_right,
                        'metadata': e.metadata
                    }
                    for e in result.entities
                ],
                'processing_time': result.processing_time,
                'metadata': result.metadata
            }, f, indent=2)
        
        print(f"[{self.system_name}] Results saved to: {output_file}")
        return result
    
    def evaluate_sample(self, text: str, ground_truth: List[Dict], timeout: int = 300) -> Dict[str, Any]:
        """Evaluate the system on a single sample"""
        result = self.link_entities(text, timeout)
        return self._evaluate_linking_result(result, ground_truth)
    
    def _evaluate_linking_result(self, result: LinkingResult, ground_truth: List[Dict]) -> Dict[str, Any]:
        """Evaluate a linking result against ground truth"""
        # Convert to sets for comparison with flexible matching
        predicted = set()
        for entity in result.entities:
            # Use entity text and entity_id for matching (more flexible than exact spans)
            predicted.add((entity.mention.lower(), entity.entity_id))
        
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
