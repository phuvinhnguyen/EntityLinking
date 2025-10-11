"""
Simple baseline entity linking system
"""
import time
import sys
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from entity_database import EntityDatabase
from systems.base_system import BaseSystem, EntityLink, LinkingResult

@dataclass
class DetectedEntity:
    """Represents a detected entity in text"""
    text: str
    start_pos: int
    end_pos: int
    context_left: str
    context_right: str
    entity_type: str = "UNKNOWN"

class SimpleSystem(BaseSystem):
    """Simple baseline entity linking system using only heuristic detection and top candidate selection"""
    
    def __init__(self, config: Config = None):
        super().__init__(config)
        self.config = config or Config()
        self.entity_database = None
    
    def initialize(self, entities_path: str) -> bool:
        """Initialize the simple system with entity database"""
        try:
            print(f"[{self.system_name}] Initializing simple system...")
            
            # Initialize entity database with BM25 only
            self.entity_database = EntityDatabase(
                search_method="bm25",
                embedding_model=None
            )
            
            # Load entity database
            self._load_entity_database(entities_path)
            
            self._initialized = True
            print(f"[{self.system_name}] Initialization completed successfully")
            return True
            
        except Exception as e:
            print(f"[{self.system_name}] Initialization failed: {e}")
            return False
    
    def _load_entity_database(self, entities_path: str):
        """Load entity database from path"""
        print(f"[{self.system_name}] Loading entity database from {entities_path}...")
        
        # Try to load from Wikidata format first, then fallback to old format
        if os.path.isdir(entities_path):
            success = self.entity_database.load_from_wikidata_format(entities_path)
        else:
            success = self.entity_database.load_from_evaluation_setup(entities_path)
        
        if not success:
            raise Exception("Failed to load entity database")
        
        self.entity_database.build_index()
        
        stats = self.entity_database.get_stats()
        print(f"[{self.system_name}] Entity database loaded: {stats['total_entities']} entities")
    
    def link_entities(self, text: str, timeout: int = None) -> LinkingResult:
        """Link entities in text using simple approach"""
        if timeout is None:
            timeout = self.config.TOTAL_PROCESSING_TIMEOUT
        
        start_time = time.time()
        
        print(f"[{self.system_name}] Starting entity linking for text of length {len(text)} (timeout: {timeout}s)")
        
        # Step 1: Detect entities using heuristics
        print(f"[{self.system_name}] Step 1: Detecting entities using heuristics...")
        detected_entities = self._detect_entities_simple(text)
        print(f"[{self.system_name}] Detected {len(detected_entities)} entities")
        
        if not detected_entities:
            return LinkingResult(
                text=text,
                entities=[],
                processing_time=time.time() - start_time,
                metadata={'error': 'No entities detected'}
            )
        
        # Step 2: Link each detected entity using simple method
        print(f"[{self.system_name}] Step 2: Linking entities using simple method...")
        linked_entities = []
        
        for i, entity in enumerate(detected_entities):
            if time.time() - start_time > timeout:
                print(f"[{self.system_name}] Entity linking timeout after {timeout}s")
                break
            
            print(f"[{self.system_name}] Linking entity {i+1}/{len(detected_entities)}: '{entity.text}'")
            
            try:
                linked_entity = self._link_single_entity_simple(entity, timeout)
                if linked_entity:
                    linked_entities.append(linked_entity)
                    
            except Exception as e:
                print(f"[{self.system_name}] Error linking entity '{entity.text}': {e}")
                continue
        
        processing_time = time.time() - start_time
        print(f"[{self.system_name}] Entity linking completed in {processing_time:.2f}s")
        
        return LinkingResult(
            text=text,
            entities=linked_entities,
            processing_time=processing_time,
            metadata={
                'detected_entities': len(detected_entities),
                'linked_entities': len(linked_entities),
                'success_rate': len(linked_entities) / len(detected_entities) if detected_entities else 0
            }
        )
    
    def _detect_entities_simple(self, text: str) -> List[DetectedEntity]:
        """Simple entity detection using heuristics"""
        entities = []
        words = text.split()
        
        # Find capitalized words (potential entities)
        for i, word in enumerate(words):
            if word[0].isupper() and len(word) > 2:
                start = text.find(word)
                if start != -1:
                    end = start + len(word)
                    context_left = text[max(0, start - 50):start]
                    context_right = text[end:min(len(text), end + 50)]
                    
                    detected_entity = DetectedEntity(
                        text=word,
                        start_pos=start,
                        end_pos=end,
                        context_left=context_left,
                        context_right=context_right,
                        entity_type="UNKNOWN"
                    )
                    entities.append(detected_entity)
        
        return entities
    
    def _link_single_entity_simple(self, entity: DetectedEntity, timeout: int) -> Optional[EntityLink]:
        """Link a single entity using simple method (top candidate)"""
        # Search for candidates
        candidates = self.entity_database.search(
            entity.text, 
            top_k=self.config.TOP_K_SEARCH,
            timeout=self.config.SEARCH_TIMEOUT
        )
        
        if not candidates:
            print(f"[{self.system_name}] No candidates found for '{entity.text}'")
            return None
        
        print(f"[{self.system_name}] Found {len(candidates)} candidates for '{entity.text}'")
        
        # Simple method: return top candidate
        best_candidate = candidates[0]
        
        return EntityLink(
            mention=entity.text,
            entity_id=best_candidate['id'],
            entity_title=best_candidate['title'],
            confidence=0.7,  # Fixed confidence for simple method
            start_pos=entity.start_pos,
            end_pos=entity.end_pos,
            context_left=entity.context_left,
            context_right=entity.context_right,
            metadata={'method': 'simple_top_candidate'}
        )

def main():
    """Main function for simple system entity linking"""
    import argparse
    import json
    import os
    
    parser = argparse.ArgumentParser(description='Simple System Entity Linking')
    parser.add_argument('--input', '-i', type=str, required=True,
                       help='Input text or path to file with sentences (one per line)')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output file path (if not provided, results will be printed to console)')
    parser.add_argument('--entities_path', '-e', type=str, 
                       default='evaluation/entities/WikiEntities',
                       help='Path to entity database')
    parser.add_argument('--timeout', '-t', type=int, default=300,
                       help='Timeout in seconds for processing')
    parser.add_argument('--top_k', type=int, default=3,
                       help='Number of candidates to retrieve')
    
    args = parser.parse_args()
    
    # Setup configuration
    config = Config()
    config.TOP_K_SEARCH = args.top_k
    
    # Create and initialize system
    print("Initializing Simple System...")
    system = SimpleSystem(config)
    
    if not system.initialize(args.entities_path):
        print("Failed to initialize simple system")
        return 1
    
    # Read input
    if os.path.isfile(args.input):
        # Read from file
        with open(args.input, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
        print(f"Processing {len(texts)} sentences from file: {args.input}")
    else:
        # Single text input
        texts = [args.input]
        print(f"Processing single text input")
    
    # Process each text
    results = []
    for i, text in enumerate(texts):
        print(f"\nProcessing text {i+1}/{len(texts)}: {text[:50]}...")
        
        try:
            result = system.link_entities(text, timeout=args.timeout)
            results.append({
                'text': text,
                'entities': [
                    {
                        'mention': entity.mention,
                        'entity_id': entity.entity_id,
                        'entity_title': entity.entity_title,
                        'confidence': entity.confidence,
                        'start_pos': entity.start_pos,
                        'end_pos': entity.end_pos,
                        'context_left': entity.context_left,
                        'context_right': entity.context_right
                    }
                    for entity in result.entities
                ],
                'processing_time': result.processing_time,
                'metadata': result.metadata
            })
            
            # Print results
            print(f"  Found {len(result.entities)} entities in {result.processing_time:.2f}s")
            for entity in result.entities:
                print(f"    '{entity.mention}' -> {entity.entity_title} (confidence: {entity.confidence:.3f})")
                
        except Exception as e:
            print(f"  Error processing text: {e}")
            results.append({
                'text': text,
                'entities': [],
                'processing_time': 0,
                'metadata': {'error': str(e)}
            })
    
    # Save or print results
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {args.output}")
    else:
        print(f"\n=== FINAL RESULTS ===")
        print(json.dumps(results, indent=2, ensure_ascii=False))
    
    return 0

def test_simple_system():
    """Test the simple system"""
    print("Testing Simple System...")
    
    # Create simple system
    system = SimpleSystem()
    
    # Initialize with entity database
    entities_path = "evaluation/entities/WikiEntities"
    if not system.initialize(entities_path):
        print("Failed to initialize simple system")
        return
    
    # Test with sample text
    test_text = """
    Apple Inc. is an American multinational technology company headquartered in Cupertino, California.
    The company was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in April 1976.
    Apple is known for its innovative products like the iPhone, iPad, and Mac computers.
    """
    
    print(f"\nTesting with text: {test_text[:100]}...")
    
    # Link entities
    result = system.link_entities(test_text, timeout=120)
    
    print(f"\nResults:")
    print(f"Processing time: {result.processing_time:.2f}s")
    print(f"Linked entities: {len(result.entities)}")
    
    for entity in result.entities:
        print(f"  - '{entity.mention}' -> {entity.entity_title} (confidence: {entity.confidence:.3f})")
    
    print("\nSimple system test completed successfully!")

if __name__ == "__main__":
    main()
