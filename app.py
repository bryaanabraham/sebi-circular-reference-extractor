import streamlit as st
import json
import re
from difflib import get_close_matches, SequenceMatcher
import numpy as np
import faiss
from openai import OpenAI
from dotenv import load_dotenv
import os
import pdfplumber
import tempfile


load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def normalize(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = re.sub(r"[\/,_\-]", "", text)
    text = re.sub(r"\s+", "", text)
    return text

def pdf_to_txt(pdf_path):
    os.makedirs("txts", exist_ok=True)
    try:
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + '\n\n'

        txt_path = os.path.join(
            "txts",
            os.path.basename(pdf_path).replace(".pdf", ".txt")
        )

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)

        return txt_path, text

    except Exception as e:
        print("Error with ", pdf_path, ": ", e)
        return None, "" 

circular_extraction_tool = {
    "type": "function",
    "function": {
        "name": "extract_circular_name",
        "description": "Extract the SEBI circular name from the document text.",
        "parameters": {
            "type": "object",
            "properties": {
                "circular_name": {
                    "type": "string",
                    "description": "The exact official name or identifier of the current circular being processed, as it appears in the document text"
                }
            },
            "required": ["circular_name"]
        }
    }
}

def extract_circular_name_from_text(text):
    prompt = f"""
You are given the full text of a SEBI document.
Extract the EXACT circular name or circular number.
Return only via tool call.

Document text:
{text}
"""

    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": prompt}],
        tools=[circular_extraction_tool],
        tool_choice={"type": "function", "function": {"name": "extract_circular_name"}}
    )

    tool_call = response.choices[0].message.tool_calls[0]
    args = json.loads(tool_call.function.arguments)

    return args.get("circular_name")


def is_similar(doc_name, documents, threshold=0.3):
    matches = get_close_matches(
        doc_name,
        documents.keys(),
        n=1,
        cutoff=threshold
    )

    if not matches:
        return False, None, 0.0

    best_match = matches[0]
    score = SequenceMatcher(None, doc_name, best_match).ratio()

    return True, documents[best_match], score


@st.cache_data
def load_knowledge_graph(path="debug/knowledge_graph_reference.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
    
@st.cache_data
def load_name_map(path="debug/name_map.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

knowledge_graph = load_knowledge_graph()
name_map = load_name_map()
documents = {normalize(d): d for d in knowledge_graph.keys()}


@st.cache_resource
def load_faiss():
    index = faiss.read_index("SEBI_RAG.faiss")
    with open("rag_metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return index, metadata

index, metadata = load_faiss()


def embed_query(q):
    return client.embeddings.create(
        model="text-embedding-3-large",
        input=q
    ).data[0].embedding


def retrieve(query, k=5):
    q_emb = np.array([embed_query(query)]).astype("float32")
    distances, indices = index.search(q_emb, k)
    return [metadata[i] for i in indices[0]]


def rag_answer(query):
    contexts = retrieve(query)

    context_text = "\n\n".join(
        f"[{c['source']}]\n{c['text']}"
        for c in contexts
    )

    prompt = f"""
Answer the question using ONLY the context below.
If the answer is not present, say so clearly.

Context:

{context_text}

Question:
{query}
"""

    resp = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": prompt}]
    )

    return resp.choices[0].message.content, contexts

def retrieve_contexts(query, k=5):
    q_emb = np.array([embed_query(query)]).astype("float32")
    distances, indices = index.search(q_emb, k)
    return [metadata[i] for i in indices[0]]

def llm_answer_from_context(query, contexts):
    context_text = "\n\n".join(
        f"Source: {c['source']}\n{c['text']}"
        for c in contexts
    )

    prompt = f"""
You are a SEBI compliance assistant.

Answer the user's question in clear, natural language using ONLY the context below.
If the answer is not explicitly supported by the context, say:
"I could not find this information in the provided SEBI documents."

Context:
{context_text}

Question:
{query}
"""

    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": "You answer like a regulatory compliance expert."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content


st.set_page_config(page_title="SEBI Knowledge Assistant", layout="wide")

st.title("SEBI Knowledge Assistant")

st.subheader("📄 Upload SEBI Circular PDF")

uploaded_pdf = st.file_uploader(
    "Upload a SEBI circular PDF",
    type=["pdf"]
)

circular_name = None
pdf_text = ""

if uploaded_pdf:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_pdf.read())
        tmp_path = tmp.name

    txt_path, pdf_text = pdf_to_txt(tmp_path)

    if pdf_text:
        with st.spinner("Extracting circular name from PDF..."):
            circular_name = extract_circular_name_from_text(pdf_text)
            circular_mapped_name = name_map[circular_name]

        if circular_name and circular_mapped_name:
            st.success(f"Detected Circular: **{circular_name}:{circular_mapped_name}**")
        else:
            st.warning("Could not extract circular name from PDF.")


mode = st.radio(
    "Choose query mode:",
    ["Lookup by Circular", "Ask a Question (RAG)"]
)

if mode == "Lookup by Circular":

    if circular_name:
        similar, best_match, score = is_similar(
            normalize(circular_name),
            documents
        )
        best_match_name = name_map[best_match]
        if not similar:
            st.warning("No matching circular found. Try exact name.")
        else:
            references = knowledge_graph.get(best_match, [])
            st.success(
                f"Showing references for **{best_match}: {best_match_name}** "
                f"(match score: {score:.2f})"
            )

            if not references:
                st.info("No referenced documents found.")
            else:
                st.subheader("📚 Referenced Documents")

                for i, ref in enumerate(references, start=1):
                    doc = ref.get("document")
                    pages = ref.get("pages", [])

                    if not doc or doc == circular_name:
                        continue

                    st.markdown(f"""
            {i}. {name_map[doc]} : {doc}  
            References on Page: {", ".join(map(str, pages)) if pages else "Not specified"}
            ---
            """)


else:
    references = []
    docs_context = ""

    if circular_name:
        similar, best_match, score = is_similar(
            normalize(circular_name),
            documents
        )
        
        if not similar:
            st.warning("No matching circular found. Try exact name.")
        else:
            best_match_name = name_map[best_match]
            references = knowledge_graph.get(best_match, [])

            if references:
                st.info(f"Using references from circular: {best_match}: {best_match_name}")

                docs_context = "\n".join(
                    f"- {ref['document']} (pages {ref.get('pages', [])})"
                    for ref in references
                    if ref.get("document")
                )

    query = st.text_area(
        "Ask a question about SEBI circulars",
        placeholder="e.g. What are the compliance requirements for FPIs?"
    )

    k = 5

    if st.button("Ask") and query:
        with st.spinner("Searching SEBI documents..."):
            query = query+"/n Answer from this context. These are the circulars along with their page numbers referred in this circular: "+docs_context
            contexts = retrieve_contexts(query, k=k)

            answer = llm_answer_from_context(
                query=f"Answer about {circular_name}: "+query,
                contexts=contexts
            )

        st.subheader("🧠 Answer")
        answer = answer.replace(circular_name, circular_mapped_name+" : ("+circular_name+")")

        st.write(answer)

        # st.subheader("📚 Sources (Retrieved Documents)")
        # for c in contexts:
        #     with st.expander(c["source"]):
        #         st.write(c["text"])
