"""
Entity database module with search capabilities
"""
import json
import os
import time
import pickle
from typing import List, Dict, Any, Optional
from collections import defaultdict
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Warning: sentence-transformers not available. Using BM25 only.")

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    print("Warning: rank-bm25 not available. Using simple BM25 implementation.")

class EntityDatabase:
    """Entity database with multiple search methods"""
    
    def __init__(self, search_method: str = "bm25", embedding_model: str = "all-MiniLM-L6-v2"):
        self.entities = {}
        self.search_method = search_method
        self.embedding_model_name = embedding_model
        self.embedding_model = None
        self.embeddings = None
        self.entity_ids = []
        self.bm25_index = None
        self.corpus_texts = []
        self.timeout = 30  # seconds for operations
        
        # Initialize search method
        self._initialize_search_method()
    
    def _initialize_search_method(self):
        """Initialize the search method"""
        if self.search_method == "embedding" and SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                print(f"Loading embedding model: {self.embedding_model_name}")
                self.embedding_model = SentenceTransformer(self.embedding_model_name)
                print("Embedding model loaded successfully")
            except Exception as e:
                print(f"Error loading embedding model: {e}")
                print("Falling back to BM25")
                self.search_method = "bm25"
        elif self.search_method == "hybrid" and SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                print(f"Loading embedding model for hybrid search: {self.embedding_model_name}")
                self.embedding_model = SentenceTransformer(self.embedding_model_name)
                print("Embedding model loaded successfully")
            except Exception as e:
                print(f"Error loading embedding model: {e}")
                print("Falling back to BM25")
                self.search_method = "bm25"
    
    def add_entity(self, entity_id: str, title: str, description: str, metadata: Dict = None):
        """Add an entity to the database"""
        entity = {
            'id': entity_id,
            'title': title,
            'description': description,
            'metadata': metadata or {}
        }
        self.entities[entity_id] = entity
    
    def load_from_sample_data(self, sample_file: str, max_entities: int = None):
        """Load entities from sample.jsonl file"""
        print(f"Loading entities from {sample_file}...")
        
        if not os.path.exists(sample_file):
            print(f"Sample file {sample_file} not found. Creating sample entities.")
            self._create_sample_entities()
            return
        
        entity_count = 0
        with open(sample_file, 'r', encoding='utf-8') as f:
            for line in f:
                if max_entities and entity_count >= max_entities:
                    break
                
                try:
                    data = json.loads(line.strip())
                    labels = data.get('labels', [])
                    
                    for label in labels:
                        if max_entities and entity_count >= max_entities:
                            break
                        
                        entity_id = label.get('entity_id', f'entity_{entity_count}')
                        name = label.get('name', 'Unknown')
                        
                        # Skip NIL entities
                        if entity_id in ['<NIL>', '<NO_MAPPING>']:
                            continue
                        
                        # Create description from context
                        text = data.get('text', '')
                        description = f"Entity mentioned in: {text[:200]}..."
                        
                        self.add_entity(
                            entity_id=entity_id,
                            title=name,
                            description=description,
                            metadata={
                                'source': 'sample_data',
                                'type': label.get('type', 'UNKNOWN'),
                                'span': label.get('span', [])
                            }
                        )
                        entity_count += 1
                
                except json.JSONDecodeError as e:
                    print(f"Error parsing line: {e}")
                    continue
        
        print(f"Loaded {entity_count} entities from sample data")
        
        # If no entities loaded, create sample ones
        if entity_count == 0:
            print("No entities loaded. Creating sample entities.")
            self._create_sample_entities()
    
    def load_from_evaluation_setup(self, entities_file: str):
        """Load entities from evaluation setup"""
        print(f"Loading entities from evaluation setup: {entities_file}")
        
        if not os.path.exists(entities_file):
            print(f"Entities file {entities_file} not found.")
            return False
        
        try:
            with open(entities_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            entities_data = data.get('entities', {})
            entity_count = 0
            
            for entity_id, entity_info in entities_data.items():
                self.add_entity(
                    entity_id=entity_id,
                    title=entity_info.get('title', 'Unknown'),
                    description=entity_info.get('description', 'No description'),
                    metadata=entity_info.get('metadata', {})
                )
                entity_count += 1
            
            print(f"Loaded {entity_count} entities from evaluation setup")
            return True
            
        except Exception as e:
            print(f"Error loading entities from evaluation setup: {e}")
            return False
    
    def load_from_wikidata_format(self, entities_dir: str):
        """Load entities from Wikidata format files"""
        print(f"Loading entities from Wikidata format: {entities_dir}")
        
        if not os.path.exists(entities_dir):
            print(f"Entities directory {entities_dir} not found.")
            return False
        
        entity_count = 0
        
        try:
            # Load all JSON files in the directory
            for filename in os.listdir(entities_dir):
                if filename.endswith('.json'):
                    entity_id = filename.replace('.json', '')
                    filepath = os.path.join(entities_dir, filename)
                    
                    with open(filepath, 'r', encoding='utf-8') as f:
                        entity_data = json.load(f)
                    
                    # Extract information from Wikidata format
                    labels = entity_data.get('labels', {})
                    descriptions = entity_data.get('descriptions', {})
                    aliases = entity_data.get('aliases', {})
                    
                    # Get English label
                    title = "Unknown"
                    if 'en' in labels:
                        title = labels['en'].get('value', 'Unknown')
                    
                    # Get English description
                    description = "No description"
                    if 'en' in descriptions:
                        description = descriptions['en'].get('value', 'No description')
                    
                    # Get English aliases
                    alias_list = []
                    if 'en' in aliases:
                        alias_list = [alias.get('value', '') for alias in aliases['en']]
                    
                    # Create metadata
                    metadata = {
                        'type': entity_data.get('type', 'item'),
                        'aliases': alias_list,
                        'source': 'wikidata_format'
                    }
                    
                    self.add_entity(
                        entity_id=entity_id,
                        title=title,
                        description=description,
                        metadata=metadata
                    )
                    entity_count += 1
            
            print(f"Loaded {entity_count} entities from Wikidata format")
            return True
            
        except Exception as e:
            print(f"Error loading entities from Wikidata format: {e}")
            return False
    
    def _create_sample_entities(self):
        """Create sample entities for testing"""
        sample_entities = [
            {
                "id": "Q848117",
                "title": "Houston Astros",
                "description": "The Houston Astros are an American professional baseball team based in Houston, Texas.",
                "metadata": {"type": "sports_team", "sport": "baseball"}
            },
            {
                "id": "Q3176522", 
                "title": "Jeff Keppinger",
                "description": "Jeff Keppinger is a former American professional baseball infielder.",
                "metadata": {"type": "person", "profession": "baseball_player"}
            },
            {
                "id": "Q487999",
                "title": "Gainesville",
                "description": "Gainesville is a city in and the county seat of Alachua County, Florida.",
                "metadata": {"type": "location", "state": "Florida"}
            },
            {
                "id": "Q1853998",
                "title": "Robin Hood's Bay",
                "description": "Robin Hood's Bay is a small fishing village and a bay located in the North York Moors National Park.",
                "metadata": {"type": "location", "country": "England"}
            },
            {
                "id": "Q2201",
                "title": "Kick-Ass",
                "description": "Kick-Ass is a 2010 superhero film based on the comic book of the same name.",
                "metadata": {"type": "film", "year": 2010}
            }
        ]
        
        for entity in sample_entities:
            self.add_entity(
                entity_id=entity["id"],
                title=entity["title"],
                description=entity["description"],
                metadata=entity["metadata"]
            )
        
        print(f"Created {len(sample_entities)} sample entities")
    
    def build_index(self):
        """Build search index"""
        if not self.entities:
            print("No entities to index")
            return
        
        print(f"Building {self.search_method} index for {len(self.entities)} entities...")
        start_time = time.time()
        
        # Prepare texts for indexing
        self.entity_ids = []
        self.corpus_texts = []
        
        for entity_id, entity in self.entities.items():
            text = f"{entity['title']} {entity['description']}"
            self.entity_ids.append(entity_id)
            self.corpus_texts.append(text)
        
        # Build index based on search method
        if self.search_method == "embedding" and self.embedding_model:
            self._build_embedding_index()
        elif self.search_method == "bm25":
            self._build_bm25_index()
        elif self.search_method == "hybrid":
            self._build_embedding_index()
            self._build_bm25_index()
        
        elapsed_time = time.time() - start_time
        print(f"Index built in {elapsed_time:.2f} seconds")
    
    def _build_embedding_index(self):
        """Build embedding index"""
        if not self.embedding_model:
            return
        
        try:
            print("Computing embeddings...")
            self.embeddings = self.embedding_model.encode(self.corpus_texts)
            print(f"Embeddings computed: {self.embeddings.shape}")
        except Exception as e:
            print(f"Error computing embeddings: {e}")
            self.embeddings = None
    
    def _build_bm25_index(self):
        """Build BM25 index"""
        try:
            if BM25_AVAILABLE:
                # Use rank_bm25 library
                tokenized_corpus = [text.lower().split() for text in self.corpus_texts]
                self.bm25_index = BM25Okapi(tokenized_corpus)
            else:
                # Simple BM25 implementation
                self.bm25_index = self._build_simple_bm25()
            print("BM25 index built successfully")
        except Exception as e:
            print(f"Error building BM25 index: {e}")
            self.bm25_index = None
    
    def _build_simple_bm25(self):
        """Simple BM25 implementation"""
        # Simple term frequency based scoring
        return {"corpus": self.corpus_texts}
    
    def search(self, query: str, top_k: int = 10, timeout: int = None) -> List[Dict[str, Any]]:
        """Search for entities"""
        if not self.entities:
            return []
        
        timeout = timeout or self.timeout
        start_time = time.time()
        
        try:
            if self.search_method == "embedding" and self.embeddings is not None:
                results = self._search_embedding(query, top_k)
            elif self.search_method == "bm25":
                results = self._search_bm25(query, top_k)
            elif self.search_method == "hybrid":
                results = self._search_hybrid(query, top_k)
            else:
                results = self._search_fallback(query, top_k)
            
            # Check timeout
            if time.time() - start_time > timeout:
                print(f"Search timeout after {timeout}s")
                return results[:top_k] if results else []
            
            return results[:top_k]
            
        except Exception as e:
            print(f"Error in search: {e}")
            return self._search_fallback(query, top_k)
    
    def _search_embedding(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Search using embeddings"""
        if not self.embedding_model or self.embeddings is None:
            return []
        
        query_embedding = self.embedding_model.encode([query])
        similarities = np.dot(self.embeddings, query_embedding.T).flatten()
        
        # Get top-k most similar entities
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            entity_id = self.entity_ids[idx]
            entity = self.entities[entity_id].copy()
            entity['similarity_score'] = float(similarities[idx])
            results.append(entity)
        
        return results
    
    def _search_bm25(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Search using BM25"""
        if not self.bm25_index:
            return []
        
        query_terms = query.lower().split()
        
        if BM25_AVAILABLE:
            # Use rank_bm25 library
            scores = self.bm25_index.get_scores(query_terms)
            top_indices = np.argsort(scores)[::-1][:top_k]
        else:
            # Simple BM25 implementation
            scores = []
            for text in self.corpus_texts:
                score = 0
                text_lower = text.lower()
                for term in query_terms:
                    score += text_lower.count(term)
                scores.append(score)
            
            top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        query_lower = query.lower()
        
        # Process ALL entities for boosting, not just top indices
        for idx, entity_id in enumerate(self.entity_ids):
            entity = self.entities[entity_id].copy()
            base_score = float(scores[idx])
            
            # Boost exact matches
            title_lower = entity['title'].lower()
            if title_lower == query_lower:
                # Exact match gets highest boost
                entity['bm25_score'] = base_score + 10.0
            elif query_lower in title_lower:
                # Partial match gets medium boost
                entity['bm25_score'] = base_score + 5.0
            else:
                entity['bm25_score'] = base_score
            
            results.append(entity)
        
        # Sort by boosted scores and return top_k
        results.sort(key=lambda x: x['bm25_score'], reverse=True)
        return results[:top_k]
    
    def _search_hybrid(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Hybrid search combining embeddings and BM25"""
        embedding_results = self._search_embedding(query, top_k * 2)
        bm25_results = self._search_bm25(query, top_k * 2)
        
        # If no embedding results, fall back to BM25
        if not embedding_results:
            return bm25_results[:top_k]
        
        # Combine and re-rank results
        combined = {}
        
        # Add embedding results
        for result in embedding_results:
            entity_id = result['id']
            combined[entity_id] = result
            combined[entity_id]['embedding_score'] = result.get('similarity_score', 0)
        
        # Add BM25 results
        for result in bm25_results:
            entity_id = result['id']
            if entity_id in combined:
                combined[entity_id]['bm25_score'] = result.get('bm25_score', 0)
            else:
                combined[entity_id] = result
                combined[entity_id]['bm25_score'] = result.get('bm25_score', 0)
                combined[entity_id]['embedding_score'] = 0
        
        # Normalize scores and combine
        embedding_scores = [r.get('embedding_score', 0) for r in combined.values()]
        bm25_scores = [r.get('bm25_score', 0) for r in combined.values()]
        
        max_embedding = max(embedding_scores) if embedding_scores and max(embedding_scores) > 0 else 1
        max_bm25 = max(bm25_scores) if bm25_scores and max(bm25_scores) > 0 else 1
        
        for result in combined.values():
            embedding_score = result.get('embedding_score', 0) / max_embedding
            bm25_score = result.get('bm25_score', 0) / max_bm25
            result['hybrid_score'] = 0.7 * embedding_score + 0.3 * bm25_score
        
        # Sort by hybrid score
        results = sorted(combined.values(), key=lambda x: x['hybrid_score'], reverse=True)
        return results[:top_k]
    
    def _search_fallback(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Fallback search using simple text matching"""
        query_lower = query.lower()
        results = []
        
        for entity_id, entity in self.entities.items():
            text = f"{entity['title']} {entity['description']}".lower()
            if query_lower in text:
                entity_copy = entity.copy()
                entity_copy['fallback_score'] = text.count(query_lower)
                results.append(entity_copy)
        
        return sorted(results, key=lambda x: x['fallback_score'], reverse=True)[:top_k]
    
    def get_entity(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """Get entity by ID"""
        return self.entities.get(entity_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        return {
            'total_entities': len(self.entities),
            'search_method': self.search_method,
            'has_embeddings': self.embeddings is not None,
            'has_bm25_index': self.bm25_index is not None
        }
    
    def save(self, filepath: str):
        """Save database to file"""
        data = {
            'entities': self.entities,
            'search_method': self.search_method,
            'embedding_model_name': self.embedding_model_name
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        # Save embeddings separately if they exist
        if self.embeddings is not None:
            embedding_file = filepath.replace('.json', '_embeddings.pkl')
            with open(embedding_file, 'wb') as f:
                pickle.dump(self.embeddings, f)
    
    def load(self, filepath: str):
        """Load database from file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.entities = data['entities']
        self.search_method = data.get('search_method', 'bm25')
        self.embedding_model_name = data.get('embedding_model_name', 'all-MiniLM-L6-v2')
        
        # Load embeddings if they exist
        embedding_file = filepath.replace('.json', '_embeddings.pkl')
        if os.path.exists(embedding_file):
            with open(embedding_file, 'rb') as f:
                self.embeddings = pickle.load(f)
        
        # Rebuild index
        self.build_index()

def test_entity_database():
    """Test the entity database functionality"""
    print("Testing Entity Database...")
    
    # Test with Wikidata format first
    print("\n=== Testing Wikidata format ===")
    db = EntityDatabase(search_method="bm25")
    success = db.load_from_wikidata_format("evaluation/entities/WikiEntities")
    
    if success:
        db.build_index()
        print(f"✓ Loaded {len(db.entities)} entities from Wikidata format")
        
        # Test search with our evaluation entities
        test_queries = ["Apple", "Steve Jobs", "California", "Houston", "Microsoft", "Los Angeles"]
        
        for query in test_queries:
            print(f"\nSearching for: '{query}'")
            results = db.search(query, top_k=3)
            
            for i, result in enumerate(results):
                score = result.get('bm25_score', "N/A")
                print(f"  {i+1}. {result['title']} (ID: {result['id']}, score: {score})")
        
        # Test get entity
        if db.entities:
            entity_id = list(db.entities.keys())[0]
            entity = db.get_entity(entity_id)
            print(f"\nGet entity {entity_id}: {entity['title']}")
        
        # Test stats
        stats = db.get_stats()
        print(f"\nDatabase stats: {stats}")
    else:
        print("✗ Failed to load Wikidata format, testing with sample data")
        
        # Fallback to sample data test
        search_methods = ["bm25", "embedding", "hybrid"]
        
        for method in search_methods:
            print(f"\n=== Testing {method} search with sample data ===")
            
            # Create database
            db = EntityDatabase(search_method=method)
            db.load_from_sample_data("sample.jsonl", max_entities=50)
            db.build_index()
            
            # Test search
            queries = ["Houston Astros", "baseball player", "Gainesville", "film"]
            
            for query in queries:
                print(f"\nSearching for: '{query}'")
                results = db.search(query, top_k=3)
                
                for i, result in enumerate(results):
                    score_key = f"{method}_score" if method != "bm25" else "bm25_score"
                    if method == "hybrid":
                        score_key = "hybrid_score"
                    elif method == "embedding":
                        score_key = "similarity_score"
                    
                    score = result.get(score_key, "N/A")
                    print(f"  {i+1}. {result['title']} (score: {score})")
    
    print("\nEntity database test completed successfully!")

if __name__ == "__main__":
    test_entity_database()
