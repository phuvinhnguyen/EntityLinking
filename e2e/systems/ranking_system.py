"""
Ranking-based entity linking system following idea.md approach
"""
import time
import json
import sys
import os
import random
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from llm_client import LLMClient
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
    descriptions: List[str] = None
    confidence: float = 0.0

class RankingSystem(BaseSystem):
    """Ranking-based entity linking system following idea.md approach"""
    
    def __init__(self, config: Config = None):
        super().__init__(config)
        self.config = config or Config()
        self.llm_client = None
        self.entity_database = None
    
    def initialize(self, entities_path: str) -> bool:
        """Initialize the ranking system with entity database"""
        try:
            print(f"[{self.system_name}] Initializing ranking system...")
            
            # Initialize LLM client with HuggingFace model
            self.llm_client = LLMClient(
                model_name=self.config.LLM_MODEL,
                model_path=self.config.LLM_MODEL_PATH,
                api_delay=self.config.LLM_API_DELAY
            )
            print(f"[{self.system_name}] LLM client initialized")
            
            # Initialize entity database with BM25 only
            self.entity_database = EntityDatabase(
                search_method="bm25",  # Use BM25 only as specified
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
        """Link entities in text following idea.md approach"""
        if timeout is None:
            timeout = self.config.TOTAL_PROCESSING_TIMEOUT
        
        start_time = time.time()
        
        print(f"[{self.system_name}] Starting entity linking for text of length {len(text)} (timeout: {timeout}s)")
        
        # Step 1: Detect entities and generate descriptions
        print(f"[{self.system_name}] Step 1: Detecting entities and generating descriptions...")
        detected_entities = self._detect_entities_with_descriptions(text, timeout)
        
        if not detected_entities:
            return LinkingResult(
                text=text,
                entities=[],
                processing_time=time.time() - start_time,
                metadata={'error': 'No entities detected'}
            )
        
        # Step 2: Link each detected entity using iterative selection
        print(f"[{self.system_name}] Step 2: Linking entities using iterative selection...")
        linked_entities = []
        
        for i, entity in enumerate(detected_entities):
            if time.time() - start_time > timeout:
                print(f"[{self.system_name}] Entity linking timeout after {timeout}s")
                break
            
            print(f"[{self.system_name}] Linking entity {i+1}/{len(detected_entities)}: '{entity.text}'")
            
            try:
                linked_entity = self._link_single_entity_iterative(entity, text, timeout)
                if linked_entity:
                    linked_entities.append(linked_entity)
                
                # Add delay between entities to prevent API quota exceed
                if i < len(detected_entities) - 1:
                    time.sleep(self.config.LLM_API_DELAY)
                    
            except Exception as e:
                print(f"[{self.system_name}] Error linking entity '{entity.text}': {e}")
                continue
        
        # Step 3: Re-match low-quality entities
        print(f"[{self.system_name}] Step 3: Re-matching low-quality entities...")
        linked_entities = self._rematch_low_quality_entities(linked_entities, text, timeout)
        
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
    
    def _detect_entities_with_descriptions(self, text: str, timeout: int) -> List[DetectedEntity]:
        """Detect entities and generate descriptions using LLM"""
        start_time = time.time()
        
        # Use LLM to detect entities and generate descriptions
        prompt = f"""
Analyze the following text and identify all named entities (people, places, organizations, products, etc.).
For each entity, provide:
1. The entity text
2. The start and end character positions
3. The entity type (PERSON, ORGANIZATION, LOCATION, PRODUCT, etc.)
4. Generate 3 different descriptions of what this entity could be

Text: "{text}"

Return the results as a JSON list of objects with keys: "text", "start", "end", "type", "descriptions".
The descriptions should be 3 different ways to describe this entity for entity linking purposes.
"""
        
        try:
            messages = [{"role": "user", "content": prompt}]
            response = self.llm_client.call(messages, max_tokens=2048)
            
            # Parse response
            entities = json.loads(response)
            if not isinstance(entities, list):
                entities = []
            
            detected_entities = []
            for entity_data in entities:
                if all(key in entity_data for key in ['text', 'start', 'end']):
                    # Extract context
                    start_pos = max(0, entity_data['start'] - 50)
                    end_pos = min(len(text), entity_data['end'] + 50)
                    context_left = text[start_pos:entity_data['start']]
                    context_right = text[entity_data['end']:end_pos]
                    
                    # Get descriptions
                    descriptions = entity_data.get('descriptions', [])
                    if not descriptions:
                        descriptions = [f"Entity: {entity_data['text']}"]
                    
                    detected_entity = DetectedEntity(
                        text=entity_data['text'],
                        start_pos=entity_data['start'],
                        end_pos=entity_data['end'],
                        context_left=context_left,
                        context_right=context_right,
                        entity_type=entity_data.get('type', 'UNKNOWN'),
                        descriptions=descriptions[:3]  # Limit to 3 descriptions
                    )
                    detected_entities.append(detected_entity)
            
            print(f"[{self.system_name}] Detected {len(detected_entities)} entities with descriptions")
            return detected_entities
            
        except Exception as e:
            print(f"[{self.system_name}] Error in entity detection: {e}")
            # Fallback to simple detection
            return self._fallback_entity_detection(text)
    
    def _fallback_entity_detection(self, text: str) -> List[DetectedEntity]:
        """Fallback entity detection using simple heuristics"""
        entities = []
        words = text.split()
        
        # Simple heuristic: capitalize words might be entities
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
                        entity_type="UNKNOWN",
                        descriptions=[f"Entity: {word}"]
                    )
                    entities.append(detected_entity)
        
        return entities
    
    def _link_single_entity_iterative(self, entity: DetectedEntity, text: str, timeout: int) -> Optional[EntityLink]:
        """Link a single entity using simple iterative selection"""
        # Search for candidates using entity text
        candidates = self.entity_database.search(
            entity.text, 
            top_k=self.config.TOP_K_SEARCH,
            timeout=self.config.SEARCH_TIMEOUT
        )
        
        if not candidates:
            return None
        
        # If few candidates, select directly
        if len(candidates) <= 3:
            best_candidate = self._select_best_entity(entity, candidates)
            return self._create_entity_link(entity, best_candidate, 0.8)
        
        # Iterative selection: reduce candidates step by step
        current_candidates = candidates
        while len(current_candidates) > 3:
            current_candidates = self._reduce_candidates(entity, current_candidates)
            time.sleep(self.config.LLM_API_DELAY)
        
        # Final selection
        best_candidate = self._select_best_entity(entity, current_candidates)
        return self._create_entity_link(entity, best_candidate, 0.7)
    
    def _reduce_candidates(self, entity: DetectedEntity, candidates: List[Dict]) -> List[Dict]:
        """Reduce candidates by half using LLM selection"""
        if len(candidates) <= 3:
            return candidates
        
        # Take top 6 candidates for selection
        top_candidates = candidates[:6]
        
        prompt = f"""
Given entity "{entity.text}" in context: "{entity.context_left} {entity.text} {entity.context_right}"

Select the 3 most relevant entities from:
{chr(10).join([f"{i+1}. {c['title']}: {c['description'][:100]}..." for i, c in enumerate(top_candidates)])}

Return numbers only (e.g., "1,3,5"):
"""
        
        try:
            response = self.llm_client.call([{"role": "user", "content": prompt}], max_tokens=100)
            selected = []
            for num in response.split(','):
                try:
                    idx = int(num.strip()) - 1
                    if 0 <= idx < len(top_candidates):
                        selected.append(top_candidates[idx])
                except:
                    continue
            
            return selected if selected else top_candidates[:3]
        except:
            return top_candidates[:3]
    
    def _select_best_entity(self, entity: DetectedEntity, candidates: List[Dict]) -> Dict:
        """Select the best entity from candidates using LLM"""
        if len(candidates) == 1:
            return candidates[0]
        
        prompt = f"""
Given entity "{entity.text}" in context: "{entity.context_left} {entity.text} {entity.context_right}"

Select the best match from:
{chr(10).join([f"{i+1}. {c['title']}: {c['description'][:100]}..." for i, c in enumerate(candidates)])}

Return the number (1-{len(candidates)}):
"""
        
        try:
            response = self.llm_client.call([{"role": "user", "content": prompt}], max_tokens=50)
            idx = int(response.strip()) - 1
            if 0 <= idx < len(candidates):
                return candidates[idx]
        except:
            pass
        
        return candidates[0]  # Fallback to first candidate
    
    def _create_entity_link(self, entity: DetectedEntity, candidate: Dict, confidence: float) -> EntityLink:
        """Create EntityLink object from detected entity and candidate"""
        return EntityLink(
            mention=entity.text,
            entity_id=candidate['id'],
            entity_title=candidate['title'],
            confidence=confidence,
            start_pos=entity.start_pos,
            end_pos=entity.end_pos,
            context_left=entity.context_left,
            context_right=entity.context_right,
            metadata={
                'method': 'iterative_selection',
                'entity_type': entity.entity_type,
                'descriptions_used': len(entity.descriptions)
            }
        )
    
    def _rematch_low_quality_entities(self, linked_entities: List[EntityLink], text: str, timeout: int) -> List[EntityLink]:
        """Re-match entities with low confidence scores"""
        low_quality = [e for e in linked_entities if e.confidence < self.config.LOW_CONFIDENCE_THRESHOLD]
        
        if not low_quality:
            return linked_entities
        
        print(f"[{self.system_name}] Re-matching {len(low_quality)} low-quality entities")
        
        # Simple re-matching: just search again with entity text
        improved_entities = []
        for entity in linked_entities:
            if entity.confidence < self.config.LOW_CONFIDENCE_THRESHOLD:
                # Try simple re-search
                candidates = self.entity_database.search(entity.mention, top_k=5, timeout=10)
                if candidates:
                    # Use first candidate with higher confidence
                    entity.entity_id = candidates[0]['id']
                    entity.entity_title = candidates[0]['title']
                    entity.confidence = 0.6  # Medium confidence for re-matched
            improved_entities.append(entity)
        
        return improved_entities
    

def main():
    """Main function for ranking system entity linking"""
    import argparse
    import json
    import os
    
    parser = argparse.ArgumentParser(description='Ranking System Entity Linking')
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
    parser.add_argument('--api_delay', type=float, default=0.1,
                       help='Delay between API calls')
    
    args = parser.parse_args()
    
    # Setup configuration
    config = Config()
    config.TOP_K_SEARCH = args.top_k
    config.LLM_API_DELAY = args.api_delay
    
    # Create and initialize system
    print("Initializing Ranking System...")
    system = RankingSystem(config)
    
    if not system.initialize(args.entities_path):
        print("Failed to initialize ranking system")
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

def test_ranking_system():
    """Test the ranking system"""
    print("Testing Ranking System...")
    
    # Create ranking system
    system = RankingSystem()
    
    # Initialize with entity database
    entities_path = "evaluation/entities/WikiEntities"
    if not system.initialize(entities_path):
        print("Failed to initialize ranking system")
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
    
    print("\nRanking system test completed successfully!")

if __name__ == "__main__":
    main()
