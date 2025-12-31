## Setup
```ptyhon
python -m pip install --upgrade pip
pip install -r requirements.txt
```
## Overview
This project builds a structured knowledge graph of SEBI circulars and their inter-document references using a one-time LLM-assisted extraction pipeline, with optional RAG-based querying and a Streamlit UI for interaction.

## High-Level Approach

Pipeline:
- Extract all SEBI circular PDFs from the official website
- Convert PDFs to text
- Use an LLM to extract referenced documents and page numbers
- Store extracted data in a structured JSON knowledge graph
- Query the knowledge graph (with optional RAG for added context)

Optional Enhancements:
- RAG-based querying for deeper contextual answers
- Streamlit UI for interactive exploration

## Why This Approach?

SEBI circulars are typically updated once every 2–6 months (based on online surveys).
This design ensures that:
- LLM API calls are only required only when circulars are updated
- Once extracted, the data can be queried repeatedly without additional LLM costs
- Optional LLM calls are used only when deeper contextual reasoning is needed

This significantly reduces operational costs compared to invoking an LLM for every uploaded document.

### Detailed Workflow
### 1. Scraping Circulars & PDFs

All circulars are extracted via web scraping (scrape_pdfs.py)

*Key functions:*

`collect_circular_links` : Sends payload-based requests to retrieve HTML links for all circulars.

`get_pdf_metadata` : Handles two page formats
- PDF-based pages: Detects iframe elements, extracts the PDF URL, and stores it in a dictionary.
- Text-based pages: Scrapes the HTML text directly and writes it to a .txt file, storing the file path.

`Fallback Mechanism` : Since scraping and downloading are time-consuming operations, the metadata dictionary is saved as a string to disk. This allows recovery using ast.literal_eval() without re-running earlier steps if failures occur downstream.

`download_all_pdfs` : Downloads all identified PDFs to a local directory.

`pdf_to_text` : Converts PDFs to text using pdfplumber.
(All PDFs were text-based; in case scanned PDFs, pytesseract or EasyOCR can be used.)

### 2. PDF Analysis Using an LLM
Each extracted text file is analyzed using an LLM to identify all referenced documents within the circular.

### 3. Output Schema
The extracted data follows this schema:
```json
{
  "circular_name": [
    {
      "document": "referred_document_name",
      "pages": [1, 3, 7]
    }
  ]
}

```

### 4. Knowledge Graph Construction
*run_llm.py*: 

- Iterate through all .txt files
- Send each file to the LLM using a function call with a strict schema
- Store results in a JSON-based knowledge graph
- Model Used: gpt-4.1

### 5. Improving Accuracy & Reducing False Positives

*clean_false_positives*: 

- Circular names are extracted strictly (they always appear at the top of the document) → lets call it List A
- Referenced documents are extracted loosely → lets call it List B
- Any reference in List B that does not match List A is discarded

Why this works: Since List A contains all valid SEBI circulars, this approach minimizes both false positives and false negatives.

`is_similar` : Allows fuzzy matching between circular names.

    Example: "Circular ABCD 25/04" and "ABCD Circular 25th April"

These should resolve to the same document despite naming inconsistencies between filenames and in-document references.

`canonicalize_reference_names`
Normalizes similar names so only a single canonical form exists across the JSON.The final cleaned data is dumped into a consolidated knowledge_graph.json.

### 6. Optional RAG-Based Contextual Querying

A Retrieval-Augmented Generation (RAG) pipeline is provided to add contextual depth when querying references.

Reference:See `debug/rag.ipynb` for implementation details.

### 7. Streamlit UI

An interactive Streamlit application is provided for querying the knowledge graph.

Run:
```python
streamlit run app.py
```

Features:
- Upload a circular PDF
- Choose between:
    - Circular Lookup (default)
    - Extracts the filename
- Queries the knowledge graph directly
    - RAG-Based Lookup
    - Runs a RAG query
    - Combines RAG output with JSON results
    - Sends both to an LLM to generate a contextual response

### Additional Resources

For a deeper understanding of the code and intermediate experiments, refer to the Jupyter notebooks in the archive/ directory.

### Models Used
- LLM: OpenAI gpt-4.1
- Embeddings: text-embedding-3-large (for RAG)

### Evaluation

Evaluation required a human-in-the-loop to monitor the `clean_false_positives` function during runtime.
This significantly improved extraction accuracy.

That said, there is scope for a more robust evaluation framework that does not rely on rule-based validation of LLM outputs.

### Limitations

- The current system relies heavily on web scraping, which is not a scalable or production-grade solution. 
- A backend integration with SEBI’s internal systems would:
    - Provide reliable access to all documents
    - Enable change detection
    - Automatically trigger reprocessing when circulars are updated

### Final Notes

This is Version 1 of the system and has been scaled to cover all SEBI circulars available on the website.
However, since the ingestion pipeline is scraping-based, there is still room for improvement in:
- Circular name accuracy
- Link extraction reliability
- Overall robustness and scalability