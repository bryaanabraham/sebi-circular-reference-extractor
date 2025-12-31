import os
import re
import json
import time
from collections import defaultdict
from difflib import get_close_matches, SequenceMatcher
from tqdm import tqdm
from openai import OpenAI
import networkx as nx
import matplotlib.pyplot as plt


def normalize(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = re.sub(r"[\/,_\-]", "", text)
    text = re.sub(r"\s+", "", text)
    return text


def is_similar(query,candidates,threshold = 0.3):
    """
    candidates: {normalized_name: original_name}
    """
    matches = get_close_matches(query, candidates.keys(), n=1, cutoff=threshold)

    if not matches:
        return False, None, 0.0

    best = matches[0]
    score = SequenceMatcher(None, query, best).ratio()
    return True, candidates[best], score


reference_extraction_tool = {
    "type": "function",
    "name": "extract_references",
    "description": "Extract the current circular name and all explicitly referenced documents with page numbers",
    "parameters": {
        "type": "object",
        "properties": {
            "circular": {
                "type": "string",
                "description": "Exact name or identifier of the current circular"
            },
            "references": {
                "type": ["array", "null"],
                "items": {
                    "type": "object",
                    "properties": {
                        "document": {"type": "string"},
                        "pages": {
                            "type": "array",
                            "items": {"type": "integer"}
                        }
                    },
                    "required": ["document", "pages"]
                }
            }
        },
        "required": ["circular", "references"]
    }
}


def extract_references_from_text(client, text):

    response = client.responses.create(
        model="gpt-4o-mini",
        input=[
            {
                "role": "system",
                "content": "You extract references of documents from a given document."
            },
            {
                "role": "user",
                "content": f"""
You MUST call the extract_references function.

Rules:
- Always call the function
- If no references exist, set references = null
- Do NOT output text
- Do NOT explain anything

Document:
{text}
"""
            }
        ],
        tools=[reference_extraction_tool],
        tool_choice={"type": "function", "name": "extract_references"}
    )

    for item in response.output:
        if item.type == "function_call" and item.name == "extract_references":
            return json.loads(item.arguments)

    raise RuntimeError("No function output returned by model")


def build_knowledge_graph(results):
    graph = defaultdict(list)
    for r in results:
        graph[r["circular"]].extend(r["references"] or [])
    return dict(graph)


def clean_false_positives(knowledge_graph, threshold= 0.3):

    documents = {
        normalize(doc): doc
        for doc in knowledge_graph.keys()
        if doc
    }

    cleaned = {}

    for circular, refs in knowledge_graph.items():
        valid_refs = []
        for ref in refs:
            norm = normalize(ref.get("document", ""))
            similar, _, _ = is_similar(norm, documents, threshold)
            if similar:
                valid_refs.append(ref)
        cleaned[circular] = valid_refs

    return cleaned


def canonicalize_reference_names(knowledge_graph):

    documents = {
        normalize(doc): doc
        for doc in knowledge_graph.keys()
        if doc
    }

    for refs in knowledge_graph.values():
        for ref in refs:
            norm = normalize(ref.get("document"))
            similar, canonical, _ = is_similar(norm, documents)
            if similar:
                ref["document"] = canonical

    return knowledge_graph


def build_nx_graph(knowledge_graph):
    G = nx.DiGraph()

    for circular, refs in knowledge_graph.items():
        if not circular:
            continue

        G.add_node(circular, type="circular")

        for ref in refs:
            doc = ref.get("document")
            pages = ref.get("pages", [])

            if not doc:
                continue

            G.add_node(doc, type="reference")
            G.add_edge(circular, doc, pages=",".join(map(str, pages)))

    return G


def visualize_graph(G: nx.DiGraph):
    plt.figure(figsize=(18, 18))
    pos = nx.spring_layout(G, k=0.6, seed=42)

    nx.draw(
        G,
        pos,
        node_size=400,
        font_size=7,
        with_labels=True,
        arrows=True
    )

    edge_labels = nx.get_edge_attributes(G, "pages")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6)

    plt.show()


if __name__ == "__main__":

    output_dir = "debug"
    files_dir = "files"

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    os.makedirs(output_dir, exist_ok=True)

    results = []

    for file in tqdm(os.listdir(files_dir)):
        if not file.endswith(".txt"):
            continue

        path = os.path.join(files_dir, file)

        try:
            with open(path, encoding="utf-8", errors="ignore") as f:
                text = f.read()

            result = extract_references_from_text(client, text)
            results.append(result)

            time.sleep(0.4)

        except Exception as e:
            print(f"Error processing {file}: {e}")

    knowledge_graph = build_knowledge_graph(results)

    with open(f"{output_dir}/knowledge_graph_raw.json", "w", encoding="utf-8") as f:
        json.dump(knowledge_graph, f, indent=2, ensure_ascii=False)

    knowledge_graph = clean_false_positives(knowledge_graph)
    knowledge_graph = canonicalize_reference_names(knowledge_graph)

    with open(f"{output_dir}/knowledge_graph_final.json", "w", encoding="utf-8") as f:
        json.dump(knowledge_graph, f, indent=2, ensure_ascii=False)

    # G = build_nx_graph(knowledge_graph)
    # visualize_graph(G)
