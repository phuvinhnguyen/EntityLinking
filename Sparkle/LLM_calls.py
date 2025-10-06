from transformers import AutoModelForCausalLM, AutoTokenizer
import os, random, time
from typing import List
from litellm import completion

def load_llm(model_name, model_path=None):
    if model_path:
        try:
            model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto')
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            return {'type': 'hf', 'model': model, 'tokenizer': tokenizer}
        except Exception as e:
            print(f"Error: {e}")
            return None
    return {'type': 'api', 'model_name': model_name}

def llm_call(messages, max_new_tokens=1024, **config):
    if config['type'] == 'hf':
        tok = config['tokenizer']
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(text, return_tensors="pt").to(config['model'].device)
        outputs = config['model'].generate(**inputs, max_new_tokens=max_new_tokens)
        return tok.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    else:
        if isinstance(messages, str): messages = [{"role": "user", "content": messages}]
        error = ''
        
        # Check if API token is available
        api_token = os.getenv('LLM_API_TOKEN')
        if not api_token:
            return "No API token available. Please set LLM_API_TOKEN environment variable."
        
        api_keys = api_token.split(',')
        for _ in range(7):
            try:
                time.sleep(7)
                response = completion(model=config['model_name'], messages=messages, max_tokens=max_new_tokens, api_key=random.choice(api_keys))
                return response.choices[0].message.content
            except Exception as e:
                error = str(e)
                continue
        return error

if __name__ == "__main__":
    messages = "What is your favourite condiment?"
    
    # API usage
    config = load_llm('gemini/gemini-2.0-flash-exp')
    print(llm_call(messages, **config))
    
    # HuggingFace usage
    # config = load_llm('Llama', 'path/to/model')
    # print(llm_call(messages, **config))