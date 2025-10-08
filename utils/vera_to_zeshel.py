import json

input_file = './data/publications.jsonl'
output_file = './data/publications_converted.jsonl'

count = 0

with open(input_file, 'r', encoding='utf-8') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:
    for line in f_in:
        line = line.strip()
        
        if line.endswith(','):
            line = line[:-1]
            
        pub = json.loads(line)
        document_id = pub['_id']['$oid']
        catalogue_title = pub['actdisease-volume-descriptor'].get('CatalogueTitle', '')
        description = pub.get('description', '')
        
        # Clean up title by removing extra quotes
        if isinstance(catalogue_title, str):
            catalogue_title = catalogue_title.strip('"')
        
        converted = {
            "document_id": document_id,
            "title": catalogue_title,
            "text": description
        }
        
        f_out.write(json.dumps(converted, ensure_ascii=False) + '\n')
        count += 1
        
        if count % 1000 == 0:
            print(f"Processed {count} publications...")

print(f"\nConverted {count} publications")
print(f"Output saved to {output_file}")

