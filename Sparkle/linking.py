from core import *
from argparse import ArgumentParser
import os, json
from collections import defaultdict

if __name__ == "__main__":
    parser = ArgumentParser()

    # Database arguments
    parser.add_argument("--documents_dir", type=str, default='/home/kat/Desktop/UppsalaUniversity/Project/EntityLinking/zeshel/documents')
    parser.add_argument("--max_documents", type=int, default=1000)

    # LLM arguments
    parser.add_argument("--model_name", type=str, default="gemini/gemma-3-4b-it")
    parser.add_argument("--model_path", type=str, default=None)

    # Input and output arguments
    parser.add_argument("--mentions_file", type=str, default=None) # mentions file path (alias for input)
    parser.add_argument("--output_file", type=str, default=None)

    # Ranking algorithm arguments
    parser.add_argument("--ranking_algorithm", type=str, default='bradley_terry')
    parser.add_argument("--n_queries", type=int, default=3)
    parser.add_argument("--top_k_search", type=int, default=20)
    parser.add_argument("--n_experiments", type=int, default=5)
    parser.add_argument("--experiment_winners", type=int, default=2)
    parser.add_argument("--experiment_subset_size", type=int, default=8)

    args = parser.parse_args()

    # Load LLM config (API or HF depending on args)
    llm_cfg = load_llm(args.model_name, args.model_path)
    entities_database = EntityDatabase(zeshel_documents_path=args.documents_dir, max_documents=args.max_documents)

    datasets = {}
    for file in os.listdir(args.documents_dir):
        with open(f'{args.documents_dir}/{file}') as f:
            data = {(obj := json.loads(line))['document_id']: obj for line in f}
            datasets[file.split('.')[0]] = data

    with open(args.mentions_file, 'r') as f:
        batch_items = []
        batch_entities = []
        batch_left_contexts = []
        batch_right_contexts = []
        batch_metadatas = []
        for i, line in enumerate(f):
            mention = json.loads(line)
            corpus = mention['corpus'] # Use this for inference
            mention_id = mention['mention_id']
            context_document_id = mention['context_document_id'] # Use this for inference
            label_document_id = mention['label_document_id']
            start_index = mention['start_index'] # Use this for inference
            end_index = mention['end_index'] # Use this for inference
            text = mention['text'] # Use this for inference

            entity = datasets[corpus][context_document_id]
            left_context = ' '.join(entity['text'].split()[:start_index])
            right_context = ' '.join(entity['text'].split()[end_index+1:])
            title = entity['title']

            target_entity = datasets[corpus][label_document_id]

            batch_entities.append(text)
            batch_left_contexts.append(left_context)
            batch_right_contexts.append(right_context)
            batch_metadatas.append({
                'target_entity': target_entity,
                'title': title,
                'corpus': corpus,
                'category': mention['category']
            })

            batch_items.append({
                'mention': mention,
                'entity': text,
                'corpus': corpus,
                'category': mention['category'],
                'title': title,
                'left_context': left_context,
                'right_context': right_context,
                'target_entity': target_entity
            })

        linking_results = linking(
            batch_entities=batch_entities,
            batch_left_contexts=batch_left_contexts,
            batch_right_contexts=batch_right_contexts,
            batch_metadatas=batch_metadatas,
            entities_database=entities_database,
            ranking_algorithm=args.ranking_algorithm,
            n_queries=args.n_queries,
            top_k_search=args.top_k_search, 
            n_experiments=args.n_experiments, 
            experiment_winners=args.experiment_winners, 
            experiment_subset_size=args.experiment_subset_size, 
            **llm_cfg)

        for i, result in enumerate(linking_results):
            batch_items[i]['linking_result'] = result

        with open(args.output_file, 'w') as f:
            for item in batch_items:
                f.write(json.dumps(item) + '\n')