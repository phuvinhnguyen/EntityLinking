from core import *
from argparse import ArgumentParser
import os, json
from collections import defaultdict

'''
Evaluation json file look like this

[
{
    "left_context": "",
    "mention": "Bandar Seri Begawan",
    "right_context": "11 15 AFP The United States today Wednesday deemed the order issued by Palestinian President Yasser Arafat for a ceasefire in territories under Palestinian Authority control as a positive gesture but considered that it does not release constitute a release form the terms of the Sharm el Sheikh agreement James Stewart the White House s spokesman in Bandar Seri Begawan the capital of the Sultanate of Brunei which American President Bill Clinton is visiting said of course we positively welcome the announcement aimed at stopping the violence But the important point is that Palestinian and Israeli officials take the right",
    "output": "Bandar Seri Begawan"
},
{
    "left_context": "Bandar Seri Begawan 11 15",
    "mention": "AFP",
    "right_context": "The United States today Wednesday deemed the order issued by Palestinian President Yasser Arafat for a ceasefire in territories under Palestinian Authority control as a positive gesture but considered that it does not release constitute a release form the terms of the Sharm el Sheikh agreement James Stewart the White House s spokesman in Bandar Seri Begawan the capital of the Sultanate of Brunei which American President Bill Clinton is visiting said of course we positively welcome the announcement aimed at stopping the violence But the important point is that Palestinian and Israeli officials take the right steps to provide",
    "output": "Agence France-Presse"
}
]

This code will

use the ner function in core.py to get the entities for each mention
if eval:
    input is json file (merge all items with same sentence together and use ner to extract entities, then each entity create a new result item)
    output is a similar json file -> save it
if inference:
    input is a list of sentences
    output is a list of entities for each sentence from ner function -> save it if output path provided

Check the ner function in core.py for more details of how it returns

'''


def _read_sentences_from_file(path:str):
    """Read sentences. If file exists: JSON array or one-per-line text; else treat input as a literal sentence."""
    if not os.path.isfile(path):
        # Treat the provided string as a single sentence
        return [path]
    with open(path, 'r') as f:
        text = f.read().strip()
        if not text:
            return []
        try:
            data = json.loads(text)
            if isinstance(data, list):
                return [str(x) for x in data]
        except Exception:
            pass
    # Fallback: treat as newline-separated
    with open(path, 'r') as f:
        return [line.strip() for line in f if line.strip()]


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="gemini/gemma-3-4b-it")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--mode", type=str, default='eval') # eval or inference
    parser.add_argument("--input", type=str, default=None) # file path for eval, or sentences file for inference
    parser.add_argument("--output_file", type=str, default=None)
    args = parser.parse_args()

    # Load LLM config (API or HF depending on args)
    llm_cfg = load_llm(args.model_name, args.model_path)

    if args.mode == 'eval':
        if not args.input:
            raise ValueError("--input is required in eval mode (path to JSON file)")
        with open(args.input, 'r') as f:
            items = json.load(f)
            if not isinstance(items, list):
                raise ValueError("Eval input must be a JSON array of items")

        # Group items by reconstructed sentence
        sentence_to_items = defaultdict(list)
        sentences = []
        for idx, it in enumerate(items):
            left = it.get('left_context', '') or ''
            mention = it.get('mention', '') or ''
            right = it.get('right_context', '') or ''
            sent = f"{left} {mention} {right}"
            sentence_to_items[sent].append(idx)
        sentences = list(sentence_to_items.keys())

        # Run NER
        ner_results = ner(sentences, **llm_cfg)

        # Build output: for every detected entity in each sentence, emit an item
        out_items = []
        for sent, ents in zip(sentences, ner_results):
            for ent in ents:
                # ent expected as [entity_text, raw_llm_output, start_index]
                try:
                    entity_text = ent[0]
                except Exception:
                    entity_text = str(ent)
                out_items.append({
                    "left_context": "",
                    "mention": entity_text,
                    "right_context": "",
                    "output": entity_text
                })

        out_path = args.output_file or (args.input + ".pred.json")
        with open(out_path, 'w') as f:
            json.dump(out_items, f, ensure_ascii=False, indent=2)
        print(f"Saved eval predictions to {out_path}")

    elif args.mode == 'inference':
        if not args.input:
            raise ValueError("--input is required in inference mode (file with sentences)")
        sentences = _read_sentences_from_file(args.input)
        if not sentences:
            print("No sentences to process.")
            exit(0)
        ner_results = ner(sentences, **llm_cfg)
        # Convert to list of entity strings per sentence
        simplified = [[(e[0] if isinstance(e, (list, tuple)) and e else str(e)) for e in ent_list] for ent_list in ner_results]

        if args.output_file:
            with open(args.output_file, 'w') as f:
                json.dump(simplified, f, ensure_ascii=False, indent=2)
            print(f"Saved inference results to {args.output_file}")
        else:
            print(json.dumps(simplified, ensure_ascii=False))
    else:
        raise ValueError("--mode must be either 'eval' or 'inference'")
