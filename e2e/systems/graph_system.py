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
        self.T_MAX = 5  # Maximum entities per LLM selection
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
        
        # If no entities detected, use fallback
        if not self.graph.nodes:
            print(f"[{self.system_name}] No entities detected, using fallback detection...")
            self._fallback_entity_detection_full(text)
        
        # Step 2: Generate descriptions for detected entities
        print(f"[{self.system_name}] Step 2: Generating descriptions for {len(self.graph.nodes)} entities...")
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
        """Build entity graph from text using LLM with context-based detection"""
        try:
            # Use LLM to detect entities and relations without requiring positions
            prompt = f"""
TEXT: {text}

INSTRUCTIONS:
1. Identify all important entities (people, organizations, locations, products, events, etc.)
2. For each entity, provide the entity text and a short context window around it
3. Identify relationships between entities

OUTPUT FORMAT:
For each entity, output:
ENTITY: [entity_text] | [context_window]

For each relation, output:
RELATION: [entity1_text] -> [entity2_text] | [relation_type]

Example:
ENTITY: Apple | technology company Apple Inc. is headquartered
ENTITY: California | headquartered in Cupertino, California
RELATION: Apple -> California | located_in

IMPORTANT:
- Entity text must match exactly what appears in the text
- Context window should be 5-10 words around the entity
- Only output entities and relations, no other text
"""
            
            messages = [{"role": "user", "content": prompt}]
            response = self.llm_client.call(messages, max_tokens=1024)
            
            # Parse the response and find positions
            self._parse_entity_graph_with_context(text, response)
            
        except Exception as e:
            print(f"[{self.system_name}] Error building entity graph: {e}")
            # Use comprehensive fallback
            self._fallback_entity_detection_full(text)
    
    def _parse_entity_graph_with_context(self, text: str, response: str):
        """Parse LLM response and find entity positions using context matching"""
        lines = response.strip().split('\n')
        entities = []
        relations = []
        
        for line in lines:
            line = line.strip()
            if line.startswith('ENTITY:'):
                # Parse entity line: ENTITY: entity_text | context_window
                entity_part = line[7:].strip()
                if '|' in entity_part:
                    entity_text, context_window = entity_part.split('|', 1)
                    entity_text = entity_text.strip()
                    context_window = context_window.strip()
                    
                    # Find the entity position using context matching
                    position_info = self._find_entity_position(text, entity_text, context_window)
                    if position_info:
                        entities.append({
                            'text': entity_text,
                            'start_pos': position_info['start_pos'],
                            'end_pos': position_info['end_pos'],
                            'context_left': position_info['context_left'],
                            'context_right': position_info['context_right']
                        })
            
            elif line.startswith('RELATION:'):
                # Parse relation line: RELATION: entity1 -> entity2 | type
                relation_part = line[9:].strip()
                if '->' in relation_part and '|' in relation_part:
                    arrow_idx = relation_part.index('->')
                    pipe_idx = relation_part.index('|')
                    
                    from_entity = relation_part[:arrow_idx].strip()
                    to_entity = relation_part[arrow_idx+2:pipe_idx].strip()
                    relation_type = relation_part[pipe_idx+1:].strip()
                    
                    relations.append({
                        'from': from_entity,
                        'to': to_entity,
                        'type': relation_type
                    })
        
        # Add entities to graph
        for i, entity_data in enumerate(entities):
            node_id = f"entity_{i}"
            node = GraphNode(
                entity_text=entity_data['text'],
                start_pos=entity_data['start_pos'],
                end_pos=entity_data['end_pos'],
                context_left=entity_data['context_left'],
                context_right=entity_data['context_right']
            )
            self.graph.add_node(node_id, node)
        
        # Add relations to graph
        for relation in relations:
            from_entity = relation['from']
            to_entity = relation['to']
            
            # Find corresponding node IDs
            from_node_id = None
            to_node_id = None
            
            for node_id, node in self.graph.nodes.items():
                if node.entity_text == from_entity:
                    from_node_id = node_id
                if node.entity_text == to_entity:
                    to_node_id = node_id
            
            if from_node_id and to_node_id:
                self.graph.add_edge(from_node_id, to_node_id, relation['type'])
    
    def _find_entity_position(self, text: str, entity_text: str, context_window: str) -> Optional[Dict]:
        """Find entity position in text using entity text and context window"""
        # First, try to find exact entity match
        entity_lower = entity_text.lower()
        text_lower = text.lower()
        
        # Try exact entity match first
        start_pos = text_lower.find(entity_lower)
        if start_pos != -1:
            end_pos = start_pos + len(entity_text)
            
            # Get context around the found position
            context_left = text[max(0, start_pos - 50):start_pos]
            context_right = text[end_pos:min(len(text), end_pos + 50)]
            
            return {
                'start_pos': start_pos,
                'end_pos': end_pos,
                'context_left': context_left,
                'context_right': context_right
            }
        
        # If exact entity not found, try using context window
        context_lower = context_window.lower()
        context_pos = text_lower.find(context_lower)
        if context_pos != -1:
            # Try to find entity within the context window
            entity_in_context_pos = context_lower.find(entity_lower)
            if entity_in_context_pos != -1:
                start_pos = context_pos + entity_in_context_pos
                end_pos = start_pos + len(entity_text)
                
                # Get context around the found position
                context_left = text[max(0, start_pos - 50):start_pos]
                context_right = text[end_pos:min(len(text), end_pos + 50)]
                
                return {
                    'start_pos': start_pos,
                    'end_pos': end_pos,
                    'context_left': context_left,
                    'context_right': context_right
                }
        
        # If still not found, try fuzzy matching with words
        entity_words = entity_lower.split()
        if len(entity_words) > 1:
            # Try to find consecutive words from entity
            for i in range(len(text_lower.split()) - len(entity_words) + 1):
                text_segment = ' '.join(text_lower.split()[i:i + len(entity_words)])
                if text_segment == entity_lower:
                    # Find the actual character positions
                    words = text.split()
                    start_pos = text_lower.find(' '.join(words[i:i + len(entity_words)]))
                    end_pos = start_pos + len(' '.join(words[i:i + len(entity_words)]))
                    
                    context_left = text[max(0, start_pos - 50):start_pos]
                    context_right = text[end_pos:min(len(text), end_pos + 50)]
                    
                    return {
                        'start_pos': start_pos,
                        'end_pos': end_pos,
                        'context_left': context_left,
                        'context_right': context_right
                    }
        
        return None
    
    def _fallback_entity_detection_full(self, text: str):
        """Comprehensive fallback entity detection"""
        entities = []
        
        # Pattern-based entity detection with position finding
        patterns = [
            # Proper nouns (capitalized words)
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',
            # Organizations with common suffixes
            r'\b[A-Z][a-zA-Z]+\s+(Inc|Corp|Company|Corporation|Ltd|LLC)\b',
            # Locations
            r'\b[A-Z][a-zA-Z]+\s+(City|State|County|Country|River|Mountain)\b',
        ]
        
        for pattern in patterns:
            for match in re.finditer(pattern, text):
                entity_text = match.group()
                start_pos = match.start()
                end_pos = match.end()
                
                # Avoid duplicates
                if not any(e['text'] == entity_text and e['start_pos'] == start_pos for e in entities):
                    context_left = text[max(0, start_pos - 50):start_pos]
                    context_right = text[end_pos:min(len(text), end_pos + 50)]
                    
                    entities.append({
                        'text': entity_text,
                        'start_pos': start_pos,
                        'end_pos': end_pos,
                        'context_left': context_left,
                        'context_right': context_right
                    })
        
        # Add entities to graph
        for i, entity_data in enumerate(entities):
            node_id = f"entity_{i}"
            node = GraphNode(
                entity_text=entity_data['text'],
                start_pos=entity_data['start_pos'],
                end_pos=entity_data['end_pos'],
                context_left=entity_data['context_left'],
                context_right=entity_data['context_right']
            )
            self.graph.add_node(node_id, node)
    
    def _generate_entity_descriptions(self, timeout: int):
        """Generate descriptions for each entity using LLM with clear instructions"""
        for node_id, node in self.graph.nodes.items():
            try:
                prompt = f"""
ENTITY: "{node.entity_text}"
CONTEXT: "...{node.context_left} {node.entity_text} {node.context_right}..."

INSTRUCTIONS:
Generate {self.N_DESCRIPTIONS} different descriptions for this entity. Each description should be a short phrase that helps identify what this entity is.

OUTPUT FORMAT:
DESCRIPTION 1: [description1]
DESCRIPTION 2: [description2]
DESCRIPTION 3: [description3]

Example:
DESCRIPTION 1: A technology company
DESCRIPTION 2: A fruit company  
DESCRIPTION 3: A multinational corporation

Make each description distinct and informative.
"""
                
                messages = [{"role": "user", "content": prompt}]
                response = self.llm_client.call(messages, max_tokens=256)
                
                # Parse descriptions from response
                descriptions = self._parse_descriptions(response)
                if descriptions:
                    node.descriptions = descriptions[:self.N_DESCRIPTIONS]
                else:
                    # Fallback descriptions
                    node.descriptions = [
                        f"The entity: {node.entity_text}",
                        f"Information about {node.entity_text}",
                        f"Details regarding {node.entity_text}"
                    ]
                
                time.sleep(self.config.LLM_API_DELAY)
                
            except Exception as e:
                print(f"[{self.system_name}] Error generating descriptions for {node.entity_text}: {e}")
                node.descriptions = [f"Entity: {node.entity_text}"]
    
    def _parse_descriptions(self, response: str) -> List[str]:
        """Parse descriptions from LLM response"""
        descriptions = []
        lines = response.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            # Match patterns like "DESCRIPTION 1: text" or "1. text"
            if re.match(r'^(DESCRIPTION\s+\d+|\d+\.?)\s*:', line, re.IGNORECASE):
                # Extract text after the colon
                colon_idx = line.find(':')
                if colon_idx != -1:
                    description = line[colon_idx + 1:].strip()
                    if description:
                        descriptions.append(description)
            elif line and not line.startswith('DESCRIPTION') and len(line) > 10:
                # Consider any substantial line as a description
                descriptions.append(line)
        
        return descriptions[:self.N_DESCRIPTIONS] if descriptions else []
    
    def _search_and_match_entities(self, timeout: int):
        """Search for relevant entities in database for each graph node"""
        for node_id, node in self.graph.nodes.items():
            try:
                # Combine entity text and descriptions for search
                search_queries = [node.entity_text] + node.descriptions
                all_candidates = []
                
                for query in search_queries:
                    if len(query) < 2:  # Skip very short queries
                        continue
                        
                    candidates = self.entity_database.search(
                        query,
                        top_k=self.K_SEARCH,
                        timeout=self.config.SEARCH_TIMEOUT
                    )
                    all_candidates.extend(candidates)
                
                # Remove duplicates and sort by score
                seen_ids = set()
                unique_candidates = []
                for candidate in sorted(all_candidates, key=lambda x: x.get('score', 0), reverse=True):
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
        """Select high-confidence entities using iterative LLM selection with clear instructions"""
        for node_id, node in self.graph.nodes.items():
            if not node.candidates:
                node.metadata['selection_reason'] = 'no_candidates'
                continue
            
            try:
                current_candidates = node.candidates.copy()
                
                # If we have few candidates, try direct matching first
                if len(current_candidates) <= 3:
                    best_candidate = self._direct_match_entity(node, current_candidates)
                    if best_candidate:
                        self._assign_entity(node, best_candidate, "high_confidence", 0.9)
                        continue
                
                # Iterative selection for more candidates
                while len(current_candidates) > 1 and len(current_candidates) > self.T_MAX:
                    current_candidates = self._llm_select_candidates(node, current_candidates, timeout)
                
                # Final selection
                if len(current_candidates) == 1:
                    self._assign_entity(node, current_candidates[0], "high_confidence", 0.9)
                elif len(current_candidates) > 1:
                    best_candidate = self._llm_rank_candidates(node, current_candidates, timeout)
                    if best_candidate:
                        self._assign_entity(node, best_candidate, "high_confidence", 0.8)
                    else:
                        # Fallback to highest scoring candidate
                        best_candidate = max(current_candidates, key=lambda x: x.get('score', 0))
                        self._assign_entity(node, best_candidate, "high_confidence", 0.7)
                
                node.metadata['final_candidates'] = len(current_candidates)
                
            except Exception as e:
                print(f"[{self.system_name}] Error in high-confidence selection for {node.entity_text}: {e}")
                # Fallback to first candidate
                if node.candidates:
                    self._assign_entity(node, node.candidates[0], "high_confidence", 0.6)
    
    def _direct_match_entity(self, node: GraphNode, candidates: List[Dict]) -> Optional[Dict]:
        """Try direct matching based on entity text similarity"""
        entity_lower = node.entity_text.lower()
        
        for candidate in candidates:
            title_lower = candidate['title'].lower()
            
            # Exact match or close match
            if (entity_lower == title_lower or 
                entity_lower in title_lower or 
                title_lower in entity_lower):
                return candidate
        
        return None
    
    def _assign_entity(self, node: GraphNode, candidate: Dict, status: str, confidence: float):
        """Assign entity candidate to node"""
        node.entity_id = candidate['id']
        node.entity_title = candidate['title']
        node.confidence = confidence
        node.status = status
        node.metadata['assigned_candidate'] = candidate['title']
    
    def _llm_select_candidates(self, node: GraphNode, candidates: List[Dict], timeout: int) -> List[Dict]:
        """Use LLM to select best candidates from a list with clear instructions"""
        try:
            prompt = f"""
ENTITY: "{node.entity_text}"
CONTEXT: "...{node.context_left} {node.entity_text} {node.context_right}..."

CANDIDATES:
"""
            
            for i, candidate in enumerate(candidates[:self.T_MAX]):
                prompt += f"{i+1}. {candidate['title']} - {candidate['description'][:150]}\n"
            
            prompt += f"""
INSTRUCTIONS:
Select the {min(3, len(candidates)//2)} most relevant candidates for the entity above.

OUTPUT FORMAT:
SELECTED: 1, 3, 5

Only output the numbers of selected candidates separated by commas. No explanations.
"""
            
            messages = [{"role": "user", "content": prompt}]
            response = self.llm_client.call(messages, max_tokens=128)
            
            # Parse selected indices
            selected_indices = self._parse_selected_indices(response)
            selected_candidates = []
            
            for idx in selected_indices:
                if 1 <= idx <= len(candidates):
                    selected_candidates.append(candidates[idx-1])
            
            return selected_candidates if selected_candidates else candidates[:3]
            
        except Exception as e:
            print(f"[{self.system_name}] Error in LLM candidate selection: {e}")
            return candidates[:3]
    
    def _parse_selected_indices(self, response: str) -> List[int]:
        """Parse selected indices from LLM response"""
        indices = []
        
        # Look for patterns like "SELECTED: 1, 3, 5" or just "1, 3, 5"
        lines = response.strip().split('\n')
        for line in lines:
            line = line.strip()
            if 'SELECTED:' in line.upper():
                line = line[line.upper().index('SELECTED:') + 9:].strip()
            
            # Extract numbers
            numbers = re.findall(r'\d+', line)
            for num in numbers:
                try:
                    indices.append(int(num))
                except ValueError:
                    continue
        
        return indices
    
    def _llm_rank_candidates(self, node: GraphNode, candidates: List[Dict], timeout: int) -> Optional[Dict]:
        """Use LLM to rank candidates using overlapping lists and confidence aggregation"""
        if len(candidates) <= 1:
            return candidates[0] if candidates else None
        
        try:
            # Create overlapping groups of candidates
            groups = self._create_overlapping_groups(candidates, group_size=3, overlap=2)
            
            # Get LLM rankings for each group with confidence
            group_results = []
            for i, group in enumerate(groups):
                group_ranking = self._llm_rank_group_with_confidence(node, group, timeout)
                if group_ranking:
                    group_results.append({
                        'group_id': i,
                        'candidates': group,
                        'ranking': group_ranking
                    })
                time.sleep(0.1)  # Small delay between group requests
            
            if not group_results:
                # Fallback to highest scoring candidate
                return max(candidates, key=lambda x: x.get('score', 0))
            
            # Aggregate results using confidence-weighted scoring
            candidate_scores = self._aggregate_group_rankings(candidates, group_results)
            
            # Return candidate with highest aggregated score
            best_candidate_id = max(candidate_scores.items(), key=lambda x: x[1])[0]
            for candidate in candidates:
                if candidate['id'] == best_candidate_id:
                    return candidate
            
            # Fallback
            return max(candidates, key=lambda x: x.get('score', 0))
            
        except Exception as e:
            print(f"[{self.system_name}] Error in LLM candidate ranking: {e}")
            return max(candidates, key=lambda x: x.get('score', 0)) if candidates else None

    def _create_overlapping_groups(self, candidates: List[Dict], group_size: int = 3, overlap: int = 2) -> List[List[Dict]]:
        """Create overlapping groups of candidates"""
        groups = []
        n = len(candidates)
        
        if n <= group_size:
            return [candidates]
        
        # Create overlapping groups
        for i in range(0, n - overlap + 1, group_size - overlap):
            group = candidates[i:i + group_size]
            if len(group) >= 2:  # Only include groups with at least 2 candidates
                groups.append(group)
            
            # Stop if we've covered all candidates
            if i + group_size >= n:
                break
        
        # Ensure all candidates appear in at least one group
        candidate_coverage = set()
        for group in groups:
            for candidate in group:
                candidate_coverage.add(candidate['id'])
        
        # Add missing candidates to appropriate groups
        for candidate in candidates:
            if candidate['id'] not in candidate_coverage:
                # Add to the group where it fits best (based on similarity)
                best_group_idx = self._find_best_group_for_candidate(candidate, groups)
                if best_group_idx is not None and len(groups[best_group_idx]) < group_size + 1:
                    groups[best_group_idx].append(candidate)
        
        return groups

    def _find_best_group_for_candidate(self, candidate: Dict, groups: List[List[Dict]]) -> Optional[int]:
        """Find the best group to add a candidate based on content similarity"""
        best_group_idx = None
        best_similarity = -1
        
        candidate_text = f"{candidate['title']} {candidate['description'][:100]}".lower()
        
        for i, group in enumerate(groups):
            group_text = " ".join([f"{c['title']} {c['description'][:50]}" for c in group]).lower()
            
            # Simple text-based similarity
            similarity = self._text_similarity(candidate_text, group_text)
            if similarity > best_similarity:
                best_similarity = similarity
                best_group_idx = i
        
        return best_group_idx if best_similarity > 0.1 else None

    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity based on common words"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        common_words = words1.intersection(words2)
        return len(common_words) / max(len(words1), len(words2))

    def _llm_rank_group_with_confidence(self, node: GraphNode, group: List[Dict], timeout: int) -> List[Dict]:
        """Get LLM ranking for a group of candidates with confidence scores"""
        try:
            prompt = f"""
    ENTITY: "{node.entity_text}" 
    CONTEXT: "...{node.context_left} {node.entity_text} {node.context_right}..."

    CANDIDATES in this group:
    """
            
            for i, candidate in enumerate(group):
                prompt += f"{i+1}. {candidate['title']} - {candidate['description'][:100]}\n"
            
            prompt += f"""
    INSTRUCTIONS:
    Rank the candidates from most relevant (1) to least relevant ({len(group)}) for the entity above.
    Also provide a confidence score (0.0 to 1.0) for your top choice.

    OUTPUT FORMAT:
    RANKING: 2, 1, 3
    CONFIDENCE: 0.85

    Example for 3 candidates:
    RANKING: 2, 1, 3
    CONFIDENCE: 0.85

    This means candidate 2 is most relevant, then 1, then 3, with 85% confidence in the top choice.

    Only output the RANKING and CONFIDENCE lines. No explanations.
    """
            
            messages = [{"role": "user", "content": prompt}]
            response = self.llm_client.call(messages, max_tokens=128)
            
            return self._parse_group_ranking(response, group)
            
        except Exception as e:
            print(f"[{self.system_name}] Error in LLM group ranking: {e}")
            return []

    def _parse_group_ranking(self, response: str, group: List[Dict]) -> List[Dict]:
        """Parse group ranking and confidence from LLM response"""
        try:
            lines = response.strip().split('\n')
            ranking_line = None
            confidence_line = None
            
            for line in lines:
                line = line.strip()
                if line.startswith('RANKING:'):
                    ranking_line = line[8:].strip()
                elif line.startswith('CONFIDENCE:'):
                    confidence_line = line[11:].strip()
            
            if not ranking_line:
                return []
            
            # Parse ranking
            ranking_numbers = []
            for part in ranking_line.split(','):
                part = part.strip()
                if part.isdigit():
                    num = int(part)
                    if 1 <= num <= len(group):
                        ranking_numbers.append(num)
            
            if len(ranking_numbers) != len(group):
                return []
            
            # Parse confidence
            confidence = 0.5  # Default confidence
            if confidence_line:
                try:
                    conf_val = float(confidence_line)
                    if 0.0 <= conf_val <= 1.0:
                        confidence = conf_val
                except ValueError:
                    pass
            
            # Create ranking results
            results = []
            for rank_pos, candidate_num in enumerate(ranking_numbers):
                candidate_idx = candidate_num - 1
                if 0 <= candidate_idx < len(group):
                    # Calculate score: higher rank = higher score, weighted by confidence
                    rank_score = (len(group) - rank_pos) / len(group)  # Normalized rank score
                    weighted_score = rank_score * confidence
                    
                    results.append({
                        'candidate': group[candidate_idx],
                        'rank': rank_pos + 1,
                        'confidence': confidence if rank_pos == 0 else confidence * 0.5,  # Lower confidence for lower ranks
                        'score': weighted_score
                    })
            
            return results
            
        except Exception as e:
            print(f"[{self.system_name}] Error parsing group ranking: {e}")
            return []

    def _aggregate_group_rankings(self, all_candidates: List[Dict], group_results: List[Dict]) -> Dict[str, float]:
        """Aggregate rankings from all groups using confidence-weighted scoring"""
        candidate_scores = {candidate['id']: 0.0 for candidate in all_candidates}
        candidate_appearances = {candidate['id']: 0 for candidate in all_candidates}
        
        # Method 1: Simple confidence-weighted scoring
        for group_result in group_results:
            for ranked_item in group_result['ranking']:
                candidate_id = ranked_item['candidate']['id']
                score = ranked_item['score']
                confidence = ranked_item['confidence']
                
                # Weight score by confidence
                weighted_score = score * confidence
                candidate_scores[candidate_id] += weighted_score
                candidate_appearances[candidate_id] += 1
        
        # Normalize by number of appearances
        for candidate_id in candidate_scores:
            if candidate_appearances[candidate_id] > 0:
                candidate_scores[candidate_id] /= candidate_appearances[candidate_id]
        
        # Method 2: Apply Bradley-Terry like adjustment for head-to-head comparisons
        candidate_scores = self._apply_pairwise_adjustment(candidate_scores, group_results, all_candidates)
        
        return candidate_scores

    def _apply_pairwise_adjustment(self, base_scores: Dict[str, float], 
                                group_results: List[Dict], 
                                all_candidates: List[Dict]) -> Dict[str, float]:
        """Apply pairwise comparison adjustment to scores"""
        pairwise_wins = defaultdict(int)
        pairwise_comparisons = defaultdict(int)
        
        # Count pairwise wins
        for group_result in group_results:
            ranking = group_result['ranking']
            confidence = ranking[0]['confidence'] if ranking else 0.5
            
            for i, item1 in enumerate(ranking):
                for j, item2 in enumerate(ranking):
                    if i < j:  # item1 is ranked higher than item2
                        candidate1 = item1['candidate']['id']
                        candidate2 = item2['candidate']['id']
                        pairwise_wins[candidate1] += confidence
                        pairwise_comparisons[(candidate1, candidate2)] += 1
        
        # Adjust scores based on pairwise performance
        adjusted_scores = base_scores.copy()
        
        for candidate_id in adjusted_scores:
            win_ratio = 0
            comparison_count = 0
            
            for (c1, c2), count in pairwise_comparisons.items():
                if c1 == candidate_id:
                    win_ratio += pairwise_wins[candidate_id]
                    comparison_count += count
                elif c2 == candidate_id:
                    comparison_count += count
            
            if comparison_count > 0:
                win_rate = win_ratio / comparison_count if comparison_count > 0 else 0.5
                # Blend original score with pairwise performance
                adjusted_scores[candidate_id] = 0.7 * adjusted_scores[candidate_id] + 0.3 * win_rate
        
        return adjusted_scores
    
    def _parse_best_index(self, response: str) -> Optional[int]:
        """Parse best candidate index from LLM response"""
        # Look for patterns like "BEST: 1" or just "1"
        lines = response.strip().split('\n')
        for line in lines:
            line = line.strip()
            if 'BEST:' in line.upper():
                line = line[line.upper().index('BEST:') + 5:].strip()
            
            # Extract first number
            numbers = re.findall(r'\d+', line)
            if numbers:
                try:
                    return int(numbers[0])
                except ValueError:
                    continue
        
        return None
    
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
            context_titles = []
            for context_node_id in context_nodes:
                context_node = self.graph.nodes[context_node_id]
                if context_node.entity_id and context_node.entity_title:
                    context_titles.append(context_node.entity_title)
            
            if not context_titles:
                return
            
            # Search with enhanced context
            search_query = f"{node.entity_text} {' '.join(context_titles)}"
            candidates = self.entity_database.search(
                search_query,
                top_k=self.K_SEARCH,
                timeout=self.config.SEARCH_TIMEOUT
            )
            
            if candidates:
                # Use LLM to select best match with context
                best_candidate = self._llm_rank_candidates_with_context(node, candidates, context_titles, timeout)
                if best_candidate:
                    node.entity_id = best_candidate['id']
                    node.entity_title = best_candidate['title']
                    node.confidence = 0.7
                    node.metadata['rematched'] = True
                    node.metadata['context_used'] = len(context_titles)
            
        except Exception as e:
            print(f"[{self.system_name}] Error in context-based re-matching: {e}")
    
    def _llm_rank_candidates_with_context(self, node: GraphNode, candidates: List[Dict], context_titles: List[str], timeout: int) -> Optional[Dict]:
        """Rank candidates with context information"""
        try:
            prompt = f"""
ENTITY: "{node.entity_text}"
CONTEXT: "...{node.context_left} {node.entity_text} {node.context_right}..."
RELATED ENTITIES: {', '.join(context_titles)}

CANDIDATES:
"""
            
            for i, candidate in enumerate(candidates):
                prompt += f"{i+1}. {candidate['title']} - {candidate['description'][:150]}\n"
            
            prompt += """
INSTRUCTIONS:
Select the candidate that best matches the entity given the context and related entities.

OUTPUT FORMAT:
BEST: [number]

Only output the number of the best candidate.
"""
            
            messages = [{"role": "user", "content": prompt}]
            response = self.llm_client.call(messages, max_tokens=128)
            
            best_index = self._parse_best_index(response)
            if best_index is not None and 1 <= best_index <= len(candidates):
                return candidates[best_index-1]
            
            return candidates[0] if candidates else None
            
        except Exception as e:
            print(f"[{self.system_name}] Error in context-based ranking: {e}")
            return candidates[0] if candidates else None
    
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
    test_graph_system()