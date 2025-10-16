"""
Configuration settings for the Entity Linking system
"""
import os
import subprocess
from typing import Dict, Any
import yaml


class Config:
    """Configuration class for entity linking system"""
    
    # LLM Configuration
    LLM_MODEL = "google/gemma-3-270m"  # Using HuggingFace model to avoid API quota
    LLM_MODEL_PATH = "google/gemma-3-270m"  # Path for HuggingFace model
    LLM_MAX_TOKENS = 512  # Reduced for smaller model
    LLM_API_DELAY = 0.1  # Minimal delay for local model
    
    # Text Processing
    CHUNK_SIZE = 1000  # Characters per chunk
    CHUNK_OVERLAP = 200  # Overlap between chunks
    
    # Entity Processing
    N_ENTITY_DESCRIPTIONS = 3  # Number of descriptions to generate per entity
    N_QUERIES_PER_ENTITY = 3  # Number of search queries per entity
    
    # Search Configuration - BM25 only to prevent resource issues
    SEARCH_METHOD = "bm25"  # "bm25", "embedding", or "hybrid" - using BM25 only
    EMBEDDING_MODEL = None  # Disabled to prevent resource issues
    TOP_K_SEARCH = 3  # Number of candidates to retrieve (reduced for testing)
    TOP_K_FINAL = 3  # Number of final candidates for ranking (reduced for testing)
    
    # Ranking Configuration
    RANKING_ALGORITHM = "bradley_terry"  # "bradley_terry", "plackett_luce", "davidson"
    N_EXPERIMENTS = 1  # Minimal experiments for testing
    EXPERIMENT_WINNERS = 1  # Minimal winners for testing
    EXPERIMENT_SUBSET_SIZE = 2  # Minimal subset size for testing
    
    # Confidence Thresholds
    HIGH_CONFIDENCE_THRESHOLD = 0.8
    LOW_CONFIDENCE_THRESHOLD = 0.3
    
    # File Paths
    SAMPLE_DATA_PATH = "sample.jsonl"
    ENTITY_DATABASE_PATH = "entity_database.json"
    OUTPUT_DIR = "outputs"
    
    # Evaluation
    EVALUATION_METRICS = ["precision", "recall", "f1", "accuracy"]
    
    # Time Limits (in seconds) to prevent infinite loops
    DEFAULT_TIMEOUT = 30
    LLM_TIMEOUT = 60
    SEARCH_TIMEOUT = 10
    RANKING_TIMEOUT = 30
    EXPERIMENT_TIMEOUT = 30
    ENTITY_DETECTION_TIMEOUT = 60
    TOTAL_PROCESSING_TIMEOUT = 300
    
    @classmethod
    def get_llm_config(cls) -> Dict[str, Any]:
        """Get LLM configuration"""
        return {
            "model_name": cls.LLM_MODEL,
            "max_tokens": cls.LLM_MAX_TOKENS,
            "api_delay": cls.LLM_API_DELAY
        }
    
    @classmethod
    def get_search_config(cls) -> Dict[str, Any]:
        """Get search configuration"""
        return {
            "method": cls.SEARCH_METHOD,
            "embedding_model": cls.EMBEDDING_MODEL,
            "top_k_search": cls.TOP_K_SEARCH,
            "top_k_final": cls.TOP_K_FINAL
        }
    
    @classmethod
    def get_ranking_config(cls) -> Dict[str, Any]:
        """Get ranking configuration"""
        return {
            "algorithm": cls.RANKING_ALGORITHM,
            "n_experiments": cls.N_EXPERIMENTS,
            "experiment_winners": cls.EXPERIMENT_WINNERS,
            "experiment_subset_size": cls.EXPERIMENT_SUBSET_SIZE
        }
    
    @classmethod
    def update_from_env(cls):
        """Update configuration from environment variables"""
        if os.getenv("LLM_MODEL"):
            cls.LLM_MODEL = os.getenv("LLM_MODEL")
        if os.getenv("SEARCH_METHOD"):
            cls.SEARCH_METHOD = os.getenv("SEARCH_METHOD")
        if os.getenv("RANKING_ALGORITHM"):
            cls.RANKING_ALGORITHM = os.getenv("RANKING_ALGORITHM")

    @classmethod
    def load_from_yaml(cls, yaml_path: str) -> "Config":
        """Load configuration values from a YAML file (layered over defaults and env) into an instance.
        Does not mutate class-level defaults.
        """
        instance = cls()
        # 1) Overlay YAML on top of instance defaults
        if yaml_path and os.path.exists(yaml_path):
            with open(yaml_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            for key, value in data.items():
                if hasattr(instance, key):
                    setattr(instance, key, value)
        # 2) Overlay environment variables without mutating class attributes
        if os.getenv("LLM_MODEL"):
            instance.LLM_MODEL = os.getenv("LLM_MODEL")
        if os.getenv("SEARCH_METHOD"):
            instance.SEARCH_METHOD = os.getenv("SEARCH_METHOD")
        if os.getenv("RANKING_ALGORITHM"):
            instance.RANKING_ALGORITHM = os.getenv("RANKING_ALGORITHM")
        return instance

    def to_dict(self) -> Dict[str, Any]:
        return {k: getattr(self, k) for k in vars(self.__class__) if k.isupper()}
