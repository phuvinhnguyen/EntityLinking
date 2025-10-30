"""
Improved LLM client with better API handling and batch processing
"""
import os
import time
import random
from typing import List, Dict, Any, Optional
from litellm import completion
import json

# Optional imports for HuggingFace models
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not available. HuggingFace models will not work.")

class LLMClient:
    """Enhanced LLM client with better error handling and batch processing"""
    
    def __init__(self, model_name: str, model_path: Optional[str] = None, api_delay: float = 1.0):
        self.model_name = model_name
        self.model_path = model_path
        self.api_delay = api_delay
        self.model = None
        self.tokenizer = None
        self.type = None
        
        # Load model
        self._load_model()
    
    def _load_model(self):
        """Load the LLM model"""
        if self.model_path and TRANSFORMERS_AVAILABLE:
            def try_load(trust_remote_code: bool, use_fast: bool) -> bool:
                # Get HuggingFace token for gated models
                hf_token = os.getenv('HUGGINGFACE_TOKEN')
                if not hf_token:
                    print("Warning: No HUGGINGFACE_TOKEN found. Model may not load if gated.")
                print(
                    f"Loading HuggingFace model from {self.model_path} (trust_remote_code={trust_remote_code}, use_fast={use_fast})"
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    device_map='auto',
                    dtype='auto',
                    token=hf_token,
                    trust_remote_code=trust_remote_code,
                )
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_path,
                    token=hf_token,
                    trust_remote_code=trust_remote_code,
                    use_fast=use_fast,
                )
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                return True

            try:
                loaded = False
                # Attempt 1: default settings
                try:
                    loaded = try_load(trust_remote_code=False, use_fast=True)
                except Exception as e1:
                    print(f"Load attempt 1 failed: {type(e1).__name__}: {e1}")
                # Attempt 2: trust_remote_code=True
                if not loaded:
                    try:
                        loaded = try_load(trust_remote_code=True, use_fast=True)
                    except Exception as e2:
                        print(f"Load attempt 2 failed: {type(e2).__name__}: {e2}")
                # Attempt 3: slow tokenizer
                if not loaded:
                    try:
                        loaded = try_load(trust_remote_code=True, use_fast=False)
                    except Exception as e3:
                        print(f"Load attempt 3 failed: {type(e3).__name__}: {e3}")

                if loaded:
                    self.type = 'hf'
                    print("HuggingFace model loaded successfully")
                else:
                    print("Error loading HuggingFace model after multiple attempts.")
                    print("Falling back to API mode")
                    self.type = 'api'
            except Exception as e:
                print(f"Unexpected error loading HuggingFace model: {type(e).__name__}: {e}")
                print("Falling back to API mode")
                self.type = 'api'
        else:
            self.type = 'api'
    
    def _get_api_token(self) -> str:
        """Get API token from environment"""
        api_token = os.getenv('LLM_API_TOKEN')
        if not api_token:
            raise ValueError("No API token available. Please set LLM_API_TOKEN environment variable.")
        return api_token
    
    def _call_api(self, messages: List[Dict[str, str]], max_tokens: int = 1024) -> str:
        """Call API with retry logic"""
        api_key = self._get_api_token()
        api_keys = api_key.split(',')
        
        for attempt in range(7):
            try:
                if attempt > 0:
                    time.sleep(self.api_delay * (2 ** attempt))  # Exponential backoff
                
                response = completion(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=max_tokens,
                    api_key=random.choice(api_keys)
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"API call attempt {attempt + 1} failed: {e}")
                if attempt == 6:  # Last attempt
                    return f"Error: {str(e)}"
                continue
    
    def _call_hf(self, messages: List[Dict[str, str]], max_tokens: int = 1024) -> str:
        """Call HuggingFace model"""
        if not TRANSFORMERS_AVAILABLE:
            return "Error: transformers not available"
        
        try:
            text = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True
            )
            return response
        except Exception as e:
            return f"Error: {str(e)}"
    
    def call(self, messages: List[Dict[str, str]], max_tokens: int = 1024) -> str:
        """Make a single LLM call"""
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        
        if self.type == 'hf':
            return self._call_hf(messages, max_tokens)
        else:
            return self._call_api(messages, max_tokens)
    
    def batch_call(self, messages_list: List[List[Dict[str, str]]], max_tokens: int = 1024) -> List[str]:
        """Make multiple LLM calls with proper delays"""
        results = []
        
        for i, messages in enumerate(messages_list):
            if i > 0 and self.type == 'api':
                time.sleep(self.api_delay)
            
            result = self.call(messages, max_tokens)
            results.append(result)
            
            print(f"Completed call {i+1}/{len(messages_list)}")
        
        return results
    
    def generate_entity_descriptions(self, entity_info: Dict[str, Any], n_descriptions: int = 3) -> List[str]:
        """Generate multiple descriptions for an entity"""
        prompt = f"""
Given the following entity information:
- ID: {entity_info.get('id', 'Unknown')}
- Title: {entity_info.get('title', 'Unknown')}
- Description: {entity_info.get('description', 'No description available')}

Generate {n_descriptions} different descriptions of this entity that could be used for entity linking. 
Each description should focus on different aspects (e.g., semantic meaning, context, domain-specific information).
Make each description concise but informative.

Return the descriptions as a JSON list of strings.
"""
        
        messages = [{"role": "user", "content": prompt}]
        response = self.call(messages)
        
        try:
            # Try to parse as JSON
            descriptions = json.loads(response)
            if isinstance(descriptions, list) and len(descriptions) >= n_descriptions:
                return descriptions[:n_descriptions]
        except json.JSONDecodeError:
            pass
        
        # Fallback: split by lines or create simple descriptions
        lines = [line.strip() for line in response.split('\n') if line.strip()]
        if len(lines) >= n_descriptions:
            return lines[:n_descriptions]
        
        # Create fallback descriptions
        base_desc = entity_info.get('description', entity_info.get('title', 'Unknown entity'))
        return [f"{base_desc} (description {i+1})" for i in range(n_descriptions)]
    
    def detect_entities(self, text: str) -> List[Dict[str, Any]]:
        """Detect entities in text using LLM"""
        prompt = f"""
Analyze the following text and identify all named entities (people, places, organizations, products, etc.).
For each entity, provide:
1. The entity text
2. The start and end character positions
3. The entity type (PERSON, ORGANIZATION, LOCATION, PRODUCT, etc.)

Text: "{text}"

Return the results as a JSON list of objects with keys: "text", "start", "end", "type".
"""
        
        messages = [{"role": "user", "content": prompt}]
        response = self.call(messages)
        
        try:
            entities = json.loads(response)
            if isinstance(entities, list):
                return entities
        except json.JSONDecodeError:
            pass
        
        # Fallback: simple entity detection
        return self._fallback_entity_detection(text)
    
    def _fallback_entity_detection(self, text: str) -> List[Dict[str, Any]]:
        """Fallback entity detection using simple heuristics"""
        entities = []
        words = text.split()
        
        # Simple heuristic: capitalize words might be entities
        for i, word in enumerate(words):
            if word[0].isupper() and len(word) > 2:
                start = text.find(word)
                entities.append({
                    "text": word,
                    "start": start,
                    "end": start + len(word),
                    "type": "UNKNOWN"
                })
        
        return entities
    
    def select_best_entity(self, entity_mention: str, candidates: List[Dict[str, Any]], context: str) -> Dict[str, Any]:
        """Select the best entity from candidates using LLM"""
        if not candidates:
            return None
        
        candidate_list = ""
        for i, candidate in enumerate(candidates[:10]):  # Limit to top 10
            candidate_list += f"{i+1}. {candidate.get('title', 'Unknown')}: {candidate.get('description', 'No description')[:200]}...\n"
        
        prompt = f"""
Given the entity mention "{entity_mention}" in the context: "{context}"

Here are the candidate entities:
{candidate_list}

Select the most relevant entity for the mention. Return the number (1-{min(len(candidates), 10)}) of the best match.
If none are relevant, return "NONE".
"""
        
        messages = [{"role": "user", "content": prompt}]
        response = self.call(messages)
        
        try:
            # Try to parse the response as a number
            choice = int(response.strip())
            if 1 <= choice <= min(len(candidates), 10):
                return candidates[choice - 1]
        except ValueError:
            pass
        
        # Fallback: return the first candidate
        return candidates[0] if candidates else None

def create_llm_client(model_name: str = None, model_path: str = None) -> LLMClient:
    """Create an LLM client with default configuration"""
    if model_name is None:
        model_name = "gemini/gemini-2.0-flash-lite"
    
    return LLMClient(model_name, model_path)

# torch is imported above if available
