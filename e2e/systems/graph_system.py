"""
Graph-based entity linking system following idea.md approach
"""
import time
import json
import sys
import os
import random
import re
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from llm_client import LLMClient
from entity_database import EntityDatabase
from systems.base_system import BaseSystem, EntityLink, LinkingResult

@dataclass
class GraphNode:
    """Represents a node in the entity graph"""
    entity_text: str
    start_pos: int
    end_pos: int
    context_left: str
    context_right: str
    descriptions: List[str] = field(default_factory=list)
    entity_id: Optional[str] = None
    entity_title: Optional[str] = None
    confidence: float = 0.0
    status: str = "pending"  # pending, high_confidence, done
    candidates: List[Dict] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

@dataclass
class GraphEdge:
    """Represents an edge (relation) between entities"""
    from_node: str
    to_node: str
    relation_type: str
    confidence: float = 0.0

class EntityGraph:
    """Graph structure for entities and their relations"""
    
    def __init__(self):
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: List[GraphEdge] = []
        self.node_relations: Dict[str, Set[str]] = defaultdict(set)
    
    def add_node(self, node_id: str, node: GraphNode):
        """Add a node to the graph"""
        self.nodes[node_id] = node
    
    def add_edge(self, from_node: str, to_node: str, relation_type: str, confidence: float = 0.0):
        """Add an edge between nodes"""
        edge = GraphEdge(from_node, to_node, relation_type, confidence)
        self.edges.append(edge)
        self.node_relations[from_node].add(to_node)
        self.node_relations[to_node].add(from_node)
    
    def get_neighbors(self, node_id: str) -> Set[str]:
        """Get all neighbor nodes"""
        return self.node_relations.get(node_id, set())
    
    def get_high_confidence_nodes(self) -> List[str]:
        """Get all high confidence nodes"""
        return [node_id for node_id, node in self.nodes.items() if node.status == "high_confidence"]
    
    def get_done_nodes(self) -> List[str]:
        """Get all done nodes"""
        return [node_id for node_id, node in self.nodes.items() if node.status == "done"]
    
    def get_pending_nodes(self) -> List[str]:
        """Get all pending nodes"""
        return [node_id for node_id, node in self.nodes.items() if node.status == "pending"]

class GraphSystem(BaseSystem):
    """Graph-based entity linking system following idea.md approach"""
    
    def __init__(self, config: Config = None):
        super().__init__(config)
        self.config = config or Config()
        self.llm_client = None
        self.entity_database = None
        self.graph = None
        
        # Graph-specific parameters
        self.M_DESCRIPTIONS = 3  # Number of descriptions per entity
        self.N_DESCRIPTIONS = 3  # Number of descriptions per detected entity
        self.K_SEARCH = 5  # Number of relevant entities to search
        self.T_MAX = 6  # Maximum entities per LLM selection
        self.HIGH_CONFIDENCE_THRESHOLD = 0.9
        self.LOW_CONFIDENCE_THRESHOLD = 0.5
    
    def initialize(self, entities_path: str) -> bool:
        """Initialize the graph system with entity database"""
        try:
            print(f"[{self.system_name}] Initializing graph system...")
            
            # Initialize LLM client with HuggingFace model
            self.llm_client = LLMClient(
                model_name=self.config.LLM_MODEL,
                model_path=self.config.LLM_MODEL_PATH,
                api_delay=self.config.LLM_API_DELAY
            )
            print(f"[{self.system_name}] LLM client initialized")
            
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
        """Link entities in text using graph-based approach"""
        if timeout is None:
            timeout = self.config.TOTAL_PROCESSING_TIMEOUT
        
        start_time = time.time()
        
        print(f"[{self.system_name}] Starting graph-based entity linking for text of length {len(text)} (timeout: {timeout}s)")
        
        # Initialize graph
        self.graph = EntityGraph()
        
        # Step 1: Build entity graph from text
        print(f"[{self.system_name}] Step 1: Building entity graph from text...")
        self._build_entity_graph(text, timeout)
        
        # Step 2: Generate descriptions for detected entities
        print(f"[{self.system_name}] Step 2: Generating descriptions for entities...")
        self._generate_entity_descriptions(timeout)
        
        # Step 3: Search and match entities
        print(f"[{self.system_name}] Step 3: Searching and matching entities...")
        self._search_and_match_entities(timeout)
        
        # Step 4: High-confidence entity selection
        print(f"[{self.system_name}] Step 4: High-confidence entity selection...")
        self._select_high_confidence_entities(timeout)
        
        # Step 5: Low-quality entity re-matching
        print(f"[{self.system_name}] Step 5: Low-quality entity re-matching...")
        self._rematch_low_quality_entities(timeout)
        
        # Step 6: Final assignment for remaining entities
        print(f"[{self.system_name}] Step 6: Final assignment for remaining entities...")
        self._final_entity_assignment()
        
        # Convert graph to results
        linked_entities = self._graph_to_entities()
        
        processing_time = time.time() - start_time
        print(f"[{self.system_name}] Graph-based entity linking completed in {processing_time:.2f}s")
        
        return LinkingResult(
            text=text,
            entities=linked_entities,
            processing_time=processing_time,
            metadata={
                'total_nodes': len(self.graph.nodes),
                'total_edges': len(self.graph.edges),
                'high_confidence_nodes': len(self.graph.get_high_confidence_nodes()),
                'done_nodes': len(self.graph.get_done_nodes()),
                'pending_nodes': len(self.graph.get_pending_nodes())
            }
        )
    
    def _build_entity_graph(self, text: str, timeout: int):
        """Build entity graph from text using LLM"""
        try:
            # Use LLM to detect entities and relations
            prompt = f"""
Analyze the following text and identify entities and their relationships. Return a JSON response with:
1. "entities": list of entities with their positions and context
2. "relations": list of relationships between entities

Text: {text}

Return only valid JSON.
"""
            
            messages = [{"role": "user", "content": prompt}]
            response = self.llm_client.call(messages, max_tokens=1024)
            
            # Parse LLM response
            try:
                result = json.loads(response)
                entities = result.get('entities', [])
                relations = result.get('relations', [])
            except:
                # Fallback to simple heuristic detection
                entity_texts = self._fallback_entity_detection(text)
                entities = []
                for i, entity_text in enumerate(entity_texts):
                    start_pos = text.find(entity_text)
                    entities.append({
                        'text': entity_text,
                        'start_pos': start_pos,
                        'end_pos': start_pos + len(entity_text),
                        'context_left': text[max(0, start_pos - 20):start_pos],
                        'context_right': text[start_pos + len(entity_text):min(len(text), start_pos + len(entity_text) + 20)]
                    })
                relations = []
            
            # Add entities to graph
            for i, entity_data in enumerate(entities):
                node_id = f"entity_{i}"
                node = GraphNode(
                    entity_text=entity_data.get('text', ''),
                    start_pos=entity_data.get('start_pos', 0),
                    end_pos=entity_data.get('end_pos', 0),
                    context_left=entity_data.get('context_left', ''),
                    context_right=entity_data.get('context_right', '')
                )
                self.graph.add_node(node_id, node)
            
            # Add relations to graph
            for relation in relations:
                from_entity = relation.get('from', '')
                to_entity = relation.get('to', '')
                relation_type = relation.get('type', 'related')
                
                # Find corresponding node IDs
                from_node_id = None
                to_node_id = None
                
                for node_id, node in self.graph.nodes.items():
                    if node.entity_text == from_entity:
                        from_node_id = node_id
                    if node.entity_text == to_entity:
                        to_node_id = node_id
                
                if from_node_id and to_node_id:
                    self.graph.add_edge(from_node_id, to_node_id, relation_type)
            
        except Exception as e:
            print(f"[{self.system_name}] Error building entity graph: {e}")
            # Fallback to simple detection
            entities = self._fallback_entity_detection(text)
            for i, entity_text in enumerate(entities):
                node_id = f"entity_{i}"
                node = GraphNode(
                    entity_text=entity_text,
                    start_pos=text.find(entity_text),
                    end_pos=text.find(entity_text) + len(entity_text),
                    context_left="",
                    context_right=""
                )
                self.graph.add_node(node_id, node)
    
    def _fallback_entity_detection(self, text: str) -> List[str]:
        """Fallback entity detection using heuristics"""
        entities = []
        words = text.split()
        
        for word in words:
            if word[0].isupper() and len(word) > 2:
                entities.append(word)
        
        return entities
    
    def _generate_entity_descriptions(self, timeout: int):
        """Generate descriptions for each entity using LLM"""
        for node_id, node in self.graph.nodes.items():
            try:
                prompt = f"""
Generate {self.N_DESCRIPTIONS} different descriptions for the entity "{node.entity_text}" in the context: "{node.context_left} {node.entity_text} {node.context_right}".

Each description should be a short sentence that helps identify this entity in different aspects, their meaning, word-choice should be different. Return as a JSON list of strings.

Example: ["A technology company", "A fruit company", "A multinational corporation"]
"""
                
                messages = [{"role": "user", "content": prompt}]
                response = self.llm_client.call(messages, max_tokens=512)
                
                try:
                    descriptions = json.loads(response)
                    if isinstance(descriptions, list):
                        node.descriptions = descriptions[:self.N_DESCRIPTIONS]
                    else:
                        node.descriptions = [response]
                except:
                    node.descriptions = [f"Entity: {node.entity_text}"]
                
                time.sleep(self.config.LLM_API_DELAY)
                
            except Exception as e:
                print(f"[{self.system_name}] Error generating descriptions for {node.entity_text}: {e}")
                node.descriptions = [f"Entity: {node.entity_text}"]
    
    def _search_and_match_entities(self, timeout: int):
        """Search for relevant entities in database for each graph node"""
        for node_id, node in self.graph.nodes.items():
            try:
                # Search using entity text
                candidates = self.entity_database.search(
                    node.entity_text,
                    top_k=self.K_SEARCH,
                    timeout=self.config.SEARCH_TIMEOUT
                )
                
                # Also search using descriptions
                for description in node.descriptions:
                    desc_candidates = self.entity_database.search(
                        description,
                        top_k=self.K_SEARCH,
                        timeout=self.config.SEARCH_TIMEOUT
                    )
                    candidates.extend(desc_candidates)
                
                # Remove duplicates and limit
                seen_ids = set()
                unique_candidates = []
                for candidate in candidates:
                    if candidate['id'] not in seen_ids:
                        unique_candidates.append(candidate)
                        seen_ids.add(candidate['id'])
                        if len(unique_candidates) >= self.K_SEARCH * 2:
                            break
                
                node.candidates = unique_candidates
                node.metadata['search_results'] = len(unique_candidates)
                
            except Exception as e:
                print(f"[{self.system_name}] Error searching for {node.entity_text}: {e}")
                node.candidates = []
    
    def _select_high_confidence_entities(self, timeout: int):
        """Select high-confidence entities using iterative LLM selection"""
        for node_id, node in self.graph.nodes.items():
            if not node.candidates:
                continue
            
            try:
                # Iterative selection process
                current_candidates = node.candidates.copy()
                
                while len(current_candidates) > 1 and len(current_candidates) > self.T_MAX:
                    # Select top T_MAX candidates
                    top_candidates = current_candidates[:self.T_MAX]
                    
                    # Ask LLM to select best candidates
                    selected = self._llm_select_candidates(node, top_candidates, timeout)
                    current_candidates = selected
                
                # Final selection
                if len(current_candidates) == 1:
                    best_candidate = current_candidates[0]
                    node.entity_id = best_candidate['id']
                    node.entity_title = best_candidate['title']
                    node.confidence = self.HIGH_CONFIDENCE_THRESHOLD
                    node.status = "high_confidence"
                elif len(current_candidates) > 1:
                    # Ask LLM to rank and select best
                    best_candidate = self._llm_rank_candidates(node, current_candidates, timeout)
                    if best_candidate:
                        node.entity_id = best_candidate['id']
                        node.entity_title = best_candidate['title']
                        node.confidence = 0.8
                        node.status = "high_confidence"
                
                node.metadata['final_candidates'] = len(current_candidates)
                
            except Exception as e:
                print(f"[{self.system_name}] Error in high-confidence selection for {node.entity_text}: {e}")
    
    def _llm_select_candidates(self, node: GraphNode, candidates: List[Dict], timeout: int) -> List[Dict]:
        """Use LLM to select best candidates from a list"""
        try:
            prompt = f"""
Given the entity "{node.entity_text}" in context "{node.context_left} {node.entity_text} {node.context_right}",
select the most relevant entities from the following candidates. Return the IDs of selected entities as a JSON list.

Candidates:
"""
            
            for i, candidate in enumerate(candidates):
                prompt += f"{i+1}. ID: {candidate['id']}, Title: {candidate['title']}, Description: {candidate['description'][:100]}...\n"
            
            prompt += "\nReturn only the IDs of the most relevant entities as a JSON list."
            
            messages = [{"role": "user", "content": prompt}]
            response = self.llm_client.call(messages, max_tokens=256)
            
            try:
                selected_ids = json.loads(response)
                if isinstance(selected_ids, list):
                    # Filter candidates by selected IDs
                    selected_candidates = [c for c in candidates if c['id'] in selected_ids]
                    return selected_candidates[:3]  # Limit to 3
            except:
                pass
            
            # Fallback: return top 3 candidates
            return candidates[:3]
            
        except Exception as e:
            print(f"[{self.system_name}] Error in LLM candidate selection: {e}")
            return candidates[:3]
    
    def _llm_rank_candidates(self, node: GraphNode, candidates: List[Dict], timeout: int) -> Optional[Dict]:
        """Use LLM to rank candidates and select the best one"""
        try:
            prompt = f"""
Given the entity "{node.entity_text}" in context "{node.context_left} {node.entity_text} {node.context_right}",
rank the following candidates from best to worst match. Return the ID of the best match.

Candidates:
"""
            
            for i, candidate in enumerate(candidates):
                prompt += f"{i+1}. ID: {candidate['id']}, Title: {candidate['title']}, Description: {candidate['description'][:100]}...\n"
            
            prompt += "\nReturn only the ID of the best match."
            
            messages = [{"role": "user", "content": prompt}]
            response = self.llm_client.call(messages, max_tokens=128)
            
            # Extract ID from response
            for candidate in candidates:
                if candidate['id'] in response:
                    return candidate
            
            # Fallback: return first candidate
            return candidates[0] if candidates else None
            
        except Exception as e:
            print(f"[{self.system_name}] Error in LLM candidate ranking: {e}")
            return candidates[0] if candidates else None
    
    def _rematch_low_quality_entities(self, timeout: int):
        """Re-match low-quality entities using graph context"""
        high_confidence_nodes = self.graph.get_high_confidence_nodes()
        done_nodes = self.graph.get_done_nodes()
        
        # Process neighbors of high-confidence and done nodes
        processed_nodes = set(high_confidence_nodes + done_nodes)
        
        for node_id in list(processed_nodes):
            neighbors = self.graph.get_neighbors(node_id)
            
            for neighbor_id in neighbors:
                if neighbor_id not in processed_nodes:
                    neighbor_node = self.graph.nodes[neighbor_id]
                    
                    if neighbor_node.status == "pending" and neighbor_node.confidence < self.LOW_CONFIDENCE_THRESHOLD:
                        # Re-match using context from connected nodes
                        self._rematch_with_context(neighbor_node, processed_nodes, timeout)
                        neighbor_node.status = "done"
                        processed_nodes.add(neighbor_id)
    
    def _rematch_with_context(self, node: GraphNode, context_nodes: Set[str], timeout: int):
        """Re-match entity using context from connected nodes"""
        try:
            # Gather context from connected nodes
            context_info = []
            for context_node_id in context_nodes:
                context_node = self.graph.nodes[context_node_id]
                if context_node.entity_id and context_node.entity_title:
                    context_info.append(f"{context_node.entity_text} -> {context_node.entity_title} (ID: {context_node.entity_id})")
            
            # Search with enhanced context
            search_query = f"{node.entity_text} {' '.join(context_info)}"
            candidates = self.entity_database.search(
                search_query,
                top_k=self.K_SEARCH,
                timeout=self.config.SEARCH_TIMEOUT
            )
            
            if candidates:
                # Use LLM to select best match with context
                best_candidate = self._llm_rank_candidates(node, candidates, timeout)
                if best_candidate:
                    node.entity_id = best_candidate['id']
                    node.entity_title = best_candidate['title']
                    node.confidence = 0.7
                    node.metadata['rematched'] = True
                    node.metadata['context_used'] = len(context_info)
            
        except Exception as e:
            print(f"[{self.system_name}] Error in context-based re-matching: {e}")
    
    def _final_entity_assignment(self):
        """Final assignment for remaining entities"""
        for node_id, node in self.graph.nodes.items():
            if node.status == "pending" and node.candidates:
                # Assign to highest scoring candidate
                best_candidate = node.candidates[0]
                node.entity_id = best_candidate['id']
                node.entity_title = best_candidate['title']
                node.confidence = 0.5
                node.status = "done"
                node.metadata['final_assignment'] = True
    
    def _graph_to_entities(self) -> List[EntityLink]:
        """Convert graph nodes to EntityLink objects"""
        entities = []
        
        for node_id, node in self.graph.nodes.items():
            if node.entity_id and node.entity_title:
                entity_link = EntityLink(
                    mention=node.entity_text,
                    entity_id=node.entity_id,
                    entity_title=node.entity_title,
                    confidence=node.confidence,
                    start_pos=node.start_pos,
                    end_pos=node.end_pos,
                    context_left=node.context_left,
                    context_right=node.context_right,
                    metadata={
                        'method': 'graph_based',
                        'status': node.status,
                        'descriptions_count': len(node.descriptions),
                        'candidates_count': len(node.candidates),
                        **node.metadata
                    }
                )
                entities.append(entity_link)
        
        return entities

def main():
    """Main function for graph system entity linking"""
    import argparse
    import json
    import os
    
    parser = argparse.ArgumentParser(description='Graph System Entity Linking')
    parser.add_argument('--input', '-i', type=str, required=True,
                       help='Input text or path to file with sentences (one per line)')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output file path (if not provided, results will be printed to console)')
    parser.add_argument('--entities_path', '-e', type=str, 
                       default='evaluation/entities/WikiEntities',
                       help='Path to entity database')
    parser.add_argument('--timeout', '-t', type=int, default=300,
                       help='Timeout in seconds for processing')
    parser.add_argument('--top_k', type=int, default=5,
                       help='Number of candidates to retrieve')
    parser.add_argument('--api_delay', type=float, default=0.1,
                       help='Delay between API calls')
    parser.add_argument('--m_descriptions', type=int, default=3,
                       help='Number of descriptions per entity')
    parser.add_argument('--n_descriptions', type=int, default=3,
                       help='Number of descriptions per detected entity')
    
    args = parser.parse_args()
    
    # Setup configuration
    config = Config()
    config.TOP_K_SEARCH = args.top_k
    config.LLM_API_DELAY = args.api_delay
    
    # Create and initialize system
    print("Initializing Graph System...")
    system = GraphSystem(config)
    system.M_DESCRIPTIONS = args.m_descriptions
    system.N_DESCRIPTIONS = args.n_descriptions
    
    if not system.initialize(args.entities_path):
        print("Failed to initialize graph system")
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
                        'context_right': entity.context_right,
                        'metadata': entity.metadata
                    }
                    for entity in result.entities
                ],
                'processing_time': result.processing_time,
                'metadata': result.metadata
            })
            
            # Print results
            print(f"  Found {len(result.entities)} entities in {result.processing_time:.2f}s")
            for entity in result.entities:
                print(f"    '{entity.mention}' -> {entity.entity_title} (confidence: {entity.confidence:.3f}, status: {entity.metadata.get('status', 'unknown')})")
                
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

def test_graph_system():
    """Test the graph system"""
    print("Testing Graph System...")
    
    # Create graph system
    system = GraphSystem()
    
    # Initialize with entity database
    entities_path = "evaluation/entities/WikiEntities"
    if not system.initialize(entities_path):
        print("Failed to initialize graph system")
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
    print(f"Graph metadata: {result.metadata}")
    
    for entity in result.entities:
        print(f"  - '{entity.mention}' -> {entity.entity_title} (confidence: {entity.confidence:.3f}, status: {entity.metadata.get('status', 'unknown')})")
    
    print("\nGraph system test completed successfully!")

if __name__ == "__main__":
    main()
