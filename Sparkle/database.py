import json
import random
import numpy as np
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
import pickle
import os
import glob
from collections import defaultdict

class EntityDatabase:
    def __init__(self, data_path=None, zeshel_documents_path=None, max_documents=None):
        """Initialize the entity database with fake data, file data, or Zeshel documents"""
        self.entities = {}
        self.embeddings = None
        self.embedding_model = None
        self.relations = {}  # entity_id -> [(relation_id, target_entity_id), ...]
        self.corpus_info = {}  # corpus -> document_count
        
        if zeshel_documents_path and os.path.exists(zeshel_documents_path):
            self._load_zeshel_documents(zeshel_documents_path, max_documents)
        elif data_path and os.path.exists(data_path):
            self.load_from_file(data_path)
        else:
            self._create_fake_data()
        
        self._initialize_embeddings()
    
    def _load_zeshel_documents(self, documents_path, max_documents=None):
        """Load documents from Zeshel format (JSONL files)"""
        print(f"Loading Zeshel documents from {documents_path}...")
        
        # Find all JSON files in the documents directory
        json_files = glob.glob(os.path.join(documents_path, "*.json"))
        
        total_loaded = 0
        for json_file in json_files:
            corpus_name = os.path.basename(json_file).replace('.json', '')
            print(f"Loading corpus: {corpus_name}")
            
            corpus_count = 0
            with open(json_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if max_documents and total_loaded >= max_documents:
                        break
                    
                    try:
                        doc = json.loads(line.strip())
                        entity_id = doc['document_id']
                        
                        # Create entity in our format
                        entity = {
                            'id': entity_id,
                            'title': doc['title'],
                            'description': doc['text'],
                            'metadata': {
                                'corpus': corpus_name,
                                'document_id': entity_id,
                                'text_length': len(doc['text'])
                            }
                        }
                        
                        self.entities[entity_id] = entity
                        corpus_count += 1
                        total_loaded += 1
                        
                    except json.JSONDecodeError as e:
                        print(f"Error parsing line in {json_file}: {e}")
                        continue
            
            self.corpus_info[corpus_name] = corpus_count
            print(f"  Loaded {corpus_count} documents from {corpus_name}")
            
            if max_documents and total_loaded >= max_documents:
                break
        
        print(f"Total documents loaded: {total_loaded}")
        print(f"Corpora: {list(self.corpus_info.keys())}")
    
    def _create_fake_data(self):
        """Create fake entity data for testing"""
        fake_entities = [
            {
                "id": "e1",
                "title": "Apple Inc.",
                "description": "Apple Inc. is an American multinational technology company that specializes in consumer electronics, computer software, and online services.",
                "metadata": {"type": "company", "founded": 1976, "industry": "technology"}
            },
            {
                "id": "e2", 
                "title": "Apple (fruit)",
                "description": "An apple is an edible fruit produced by an apple tree. Apples are grown worldwide and are the most widely consumed fruit.",
                "metadata": {"type": "fruit", "color": "red", "nutrition": "vitamin_c"}
            },
            {
                "id": "e3",
                "title": "Microsoft Corporation",
                "description": "Microsoft Corporation is an American multinational technology corporation which produces computer software, consumer electronics, personal computers, and related services.",
                "metadata": {"type": "company", "founded": 1975, "industry": "technology"}
            },
            {
                "id": "e4",
                "title": "Google LLC",
                "description": "Google LLC is an American multinational technology company that specializes in Internet-related services and products.",
                "metadata": {"type": "company", "founded": 1998, "industry": "technology"}
            },
            {
                "id": "e5",
                "title": "Banana",
                "description": "A banana is an elongated, edible fruit botanically a berry produced by several kinds of large herbaceous flowering plants in the genus Musa.",
                "metadata": {"type": "fruit", "color": "yellow", "nutrition": "potassium"}
            },
            {
                "id": "e6",
                "title": "iPhone",
                "description": "iPhone is a line of smartphones designed and marketed by Apple Inc. that use Apple's iOS mobile operating system.",
                "metadata": {"type": "product", "manufacturer": "Apple", "category": "smartphone"}
            },
            {
                "id": "e7",
                "title": "Windows",
                "description": "Microsoft Windows is a group of several proprietary graphical operating system families developed and marketed by Microsoft.",
                "metadata": {"type": "product", "manufacturer": "Microsoft", "category": "operating_system"}
            },
            {
                "id": "e8",
                "title": "Steve Jobs",
                "description": "Steven Paul Jobs was an American business magnate, industrial designer, investor, and media proprietor.",
                "metadata": {"type": "person", "born": 1955, "died": 2011, "profession": "entrepreneur"}
            }
        ]
        
        for entity in fake_entities:
            self.entities[entity["id"]] = entity
        
        # Add some relations
        self.add_relation("e1", "founded_by", "e8")  # Apple founded by Steve Jobs
        self.add_relation("e6", "manufactured_by", "e1")  # iPhone manufactured by Apple
        self.add_relation("e7", "developed_by", "e3")  # Windows developed by Microsoft
    
    def _initialize_embeddings(self):
        """Initialize sentence transformer for embeddings"""
        try:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            self._compute_embeddings()
        except Exception as e:
            print(f"Warning: Could not load sentence transformer: {e}")
            self.embedding_model = None
    
    def _compute_embeddings(self):
        """Compute embeddings for all entities"""
        if not self.embedding_model:
            return
        
        texts = []
        entity_ids = []
        for entity_id, entity in self.entities.items():
            text = f"{entity['title']} {entity['description']}"
            texts.append(text)
            entity_ids.append(entity_id)
        
        self.embeddings = self.embedding_model.encode(texts)
        self.entity_ids = entity_ids
    
    def embedding(self, query: str, top_k: int = 10) -> List[Dict]:
        """Find entities using semantic similarity"""
        if not self.embedding_model or self.embeddings is None:
            return self._fallback_search(query, top_k)
        
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
    
    def name(self, entity_name: str) -> List[Dict]:
        """Find entities by exact or partial name match"""
        results = []
        entity_name_lower = entity_name.lower()
        
        for entity_id, entity in self.entities.items():
            title_lower = entity['title'].lower()
            if entity_name_lower in title_lower or title_lower in entity_name_lower:
                entity_copy = entity.copy()
                entity_copy['name_match_score'] = 1.0 if entity_name_lower == title_lower else 0.7
                results.append(entity_copy)
        
        return sorted(results, key=lambda x: x['name_match_score'], reverse=True)
    
    def bm25(self, query: str, top_k: int = 10) -> List[Dict]:
        """Simple BM25-like search (simplified implementation)"""
        query_terms = query.lower().split()
        results = []
        
        for entity_id, entity in self.entities.items():
            text = f"{entity['title']} {entity['description']}".lower()
            score = 0
            
            for term in query_terms:
                if term in text:
                    # Simple term frequency scoring
                    score += text.count(term)
            
            if score > 0:
                entity_copy = entity.copy()
                entity_copy['bm25_score'] = score
                results.append(entity_copy)
        
        return sorted(results, key=lambda x: x['bm25_score'], reverse=True)[:top_k]
    
    def related(self, entity_id: str) -> List[Dict]:
        """Find entities related to the given entity"""
        if entity_id not in self.relations:
            return []
        
        related_entities = []
        for relation_id, target_entity_id in self.relations[entity_id]:
            if target_entity_id in self.entities:
                entity = self.entities[target_entity_id].copy()
                entity['relation_type'] = relation_id
                related_entities.append(entity)
        
        return related_entities
    
    def add_relation(self, entity_id1: str, relation_id: str, entity_id2: str):
        """Add a relation between two entities"""
        if entity_id1 not in self.relations:
            self.relations[entity_id1] = []
        self.relations[entity_id1].append((relation_id, entity_id2))
    
    def _fallback_search(self, query: str, top_k: int) -> List[Dict]:
        """Fallback search when embeddings are not available"""
        return self.bm25(query, top_k)
    
    def get_entity(self, entity_id: str) -> Dict:
        """Get entity by ID"""
        return self.entities.get(entity_id)
    
    def get_corpus_stats(self) -> Dict:
        """Get statistics about loaded corpora"""
        return {
            'total_entities': len(self.entities),
            'corpora': self.corpus_info,
            'total_corpora': len(self.corpus_info)
        }
    
    def get_entities_by_corpus(self, corpus_name: str) -> List[Dict]:
        """Get all entities from a specific corpus"""
        entities = []
        for entity in self.entities.values():
            if entity['metadata'].get('corpus') == corpus_name:
                entities.append(entity)
        return entities
    
    def save_to_file(self, filepath: str):
        """Save database to file"""
        data = {
            'entities': self.entities,
            'relations': self.relations
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_from_file(self, filepath: str):
        """Load database from file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        self.entities = data['entities']
        self.relations = data['relations']
        self._initialize_embeddings()

def test_database():
    """Test the database functionality"""
    print("Testing Entity Database...")
    
    # Test with fake data first
    print("\n=== Testing with Fake Data ===")
    db = EntityDatabase()
    _test_database_operations(db)
    
    # Test with Zeshel data if available
    zeshel_path = "/home/kat/Desktop/UppsalaUniversity/Project/EntityLinking/zeshel/documents"
    if os.path.exists(zeshel_path):
        print("\n=== Testing with Zeshel Data (small subset) ===")
        db_zeshel = EntityDatabase(zeshel_documents_path=zeshel_path, max_documents=100)
        _test_database_operations(db_zeshel)
        
        # Test corpus-specific functionality
        print("\n6. Testing corpus statistics:")
        stats = db_zeshel.get_corpus_stats()
        print(f"  - Total entities: {stats['total_entities']}")
        print(f"  - Total corpora: {stats['total_corpora']}")
        print(f"  - Corpora: {list(stats['corpora'].keys())}")
        
        # Test corpus-specific search
        if stats['corpora']:
            corpus_name = list(stats['corpora'].keys())[0]
            print(f"\n7. Testing entities from corpus '{corpus_name}':")
            entities = db_zeshel.get_entities_by_corpus(corpus_name)
            print(f"  - Found {len(entities)} entities")
            if entities:
                print(f"  - Example: {entities[0]['title']}")
    else:
        print(f"\nZeshel documents not found at {zeshel_path}")
    
    print("\nDatabase test completed successfully!")

def _test_database_operations(db):
    """Test common database operations"""
    # Get a sample entity ID for testing
    sample_entity_id = list(db.entities.keys())[0] if db.entities else None
    
    # Test embedding search
    print("\n1. Testing embedding search:")
    if sample_entity_id:
        # Use a more generic query that works for both fake and Zeshel data
        query = "character" if any("muppets" in str(meta) for meta in [e.get('metadata', {}) for e in db.entities.values()]) else "technology company"
        results = db.embedding(query, top_k=3)
        for result in results:
            print(f"  - {result['title']} (score: {result.get('similarity_score', 'N/A')})")
    else:
        print("  - No entities available for testing")
    
    # Test name search
    print("\n2. Testing name search:")
    if sample_entity_id:
        # Use a name that might exist in the data
        search_name = "Apple" if "e1" in db.entities else "character"
        results = db.name(search_name)
        for result in results:
            print(f"  - {result['title']} (score: {result.get('name_match_score', 'N/A')})")
        if not results:
            print("  - No exact name matches found")
    else:
        print("  - No entities available for testing")
    
    # Test BM25 search
    print("\n3. Testing BM25 search:")
    if sample_entity_id:
        # Use a query that works for both data types
        query = "character" if any("muppets" in str(meta) for meta in [e.get('metadata', {}) for e in db.entities.values()]) else "fruit apple"
        results = db.bm25(query, top_k=3)
        for result in results:
            print(f"  - {result['title']} (score: {result.get('bm25_score', 'N/A')})")
    else:
        print("  - No entities available for testing")
    
    # Test related entities
    print("\n4. Testing related entities:")
    if sample_entity_id:
        results = db.related(sample_entity_id)
        if results:
            for result in results:
                print(f"  - {result['title']} (relation: {result.get('relation_type', 'N/A')})")
        else:
            print("  - No related entities found (this is normal for Zeshel data)")
    else:
        print("  - No entities available for testing")
    
    # Test get entity
    print("\n5. Testing get entity:")
    if sample_entity_id:
        entity = db.get_entity(sample_entity_id)
        if entity:
            print(f"  - {entity['title']}: {entity['description'][:100]}...")
            # Show metadata if available
            if entity.get('metadata'):
                corpus = entity['metadata'].get('corpus', 'unknown')
                print(f"    Corpus: {corpus}")
        else:
            print(f"  - Entity {sample_entity_id} not found")
    else:
        print("  - No entities available for testing")

if __name__ == "__main__":
    test_database()
