import os, json, requests, time
from typing import List, Set, Optional
from collections import deque

WIKIDATA_API = "https://www.wikidata.org/wiki/Special:EntityData/{}.json"
def download_entity(entity_id: str, output_dir: str) -> dict:
    """Download one Wikidata entity JSON and save it locally."""
    time.sleep(0.3)
    url = WIKIDATA_API.format(entity_id)
    header = {"User-Agent": "Mozilla/5.0 (compatible; EntityDownloader/1.0; +https://yourdomain.com)"}
    response = requests.get(url, headers=header)
    response.raise_for_status()

    data = response.json()
    entity_data = data["entities"][entity_id]

    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"{entity_id}.json")

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(entity_data, f, ensure_ascii=False, indent=2)

    print(f"Saved {entity_id} to {file_path}")
    return entity_data


def extract_related_entities(entity_data: dict) -> Set[str]:
    """Extract related entity IDs from 'claims' (e.g., Qxx values)."""
    related = set()
    claims = entity_data.get("claims", {})
    for prop, claim_list in claims.items():
        for claim in claim_list:
            mainsnak = claim.get("mainsnak", {})
            datavalue = mainsnak.get("datavalue", {})
            value = datavalue.get("value", {})
            if isinstance(value, dict) and "id" in value:
                related.add(value["id"])
    return related


def download_entities(
    entity_ids: List[str],
    output_dir: str = "entities",
    include_related: bool = False,
    max_entities: Optional[int] = None,
):
    """Download multiple Wikidata entities in two phases:
    1) Download all entities from the original list first.
    2) Then, if include_related=True, expand to related entities (BFS) until max_entities.
    """
    os.makedirs(output_dir, exist_ok=True)
    visited: Set[str] = set()

    # Phase 1: Download all base entities in the given order
    related_queue: deque[str] = deque()
    for eid in entity_ids:
        if eid in visited:
            continue
        try:
            entity_data = download_entity(eid, output_dir)
            visited.add(eid)
            if include_related:
                for rid in extract_related_entities(entity_data):
                    if rid not in visited:
                        related_queue.append(rid)
        except Exception as e:
            print(f"Failed to download {eid}: {e}")

    # Phase 2: Download related entities breadth-first
    if include_related and (max_entities is None or len(visited) < max_entities):
        while related_queue:
            if max_entities is not None and len(visited) >= max_entities:
                break
            current_id = related_queue.popleft()
            if current_id in visited:
                continue
            try:
                entity_data = download_entity(current_id, output_dir)
                visited.add(current_id)
                # Enqueue further relations for BFS
                for rid in extract_related_entities(entity_data):
                    if rid not in visited:
                        related_queue.append(rid)
            except Exception as e:
                print(f"Failed to download {current_id}: {e}")

    print(f"Downloaded {len(visited)} entities total (base: {min(len(entity_ids), len(visited))}, include_related={include_related}).")


if __name__ == "__main__":
    base_entities = []
    test_data_dir = os.path.join(os.path.dirname(__file__), "../../test_data")
    test_data_dir = os.path.normpath(test_data_dir)
    for jsonl_file in os.listdir(test_data_dir):
        if jsonl_file.endswith('.jsonl'):
            with open(os.path.join(test_data_dir, jsonl_file), 'r') as f:
                for line in f:
                    data = json.loads(line)
                    for entity in data['labels']:
                        base_entities.append(entity['entity_id'])

    # Default behavior: download base first, then related with a reasonable cap
    download_entities(
        base_entities,
        output_dir="WikiEntities",
        include_related=True,
        max_entities=5000,
    )
