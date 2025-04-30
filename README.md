### Anxiety QA Platform

#### Table of Contents
1. [Overview](#overview)  
2. [Features](#features)  
3. [Architecture](#architecture)  
4. [Getting Started](#getting-started)  
   - [Prerequisites](#prerequisites)  
   - [Installation](#installation)  
   - [Configuration](#configuration)  
5. [Data Ingestion](#data-ingestion)  
   - [Clinical Guidelines](#clinical-guidelines)  
   - [PubMed Papers](#pubmed-papers)  
   - [Tree Generation Utility](#tree-generation-utility)  
6. [Usage](#usage)  
   - [CLI Q&A](#cli-qa)  
   - [FastAPI Backend](#fastapi-backend)  
   - [Streamlit Frontend](#streamlit-frontend)  
7. [Directory Structure](#directory-structure)  
8. [Data Formats](#data-formats)  
9. [Dependencies](#dependencies)  
10. [Contributing](#contributing)  
11. [License](#license)  
12. [Author & Contact](#author--contact)

---

### Overview
An end-to-end question-answering system for anxiety disorders.  
Combines:  
- PySpark MapReduce retrieval over PubMed abstracts  
- FAISS-backed vector search on clinical guideline snippets in MongoDB  
- DuckDuckGo web snippet lookup  
- LLM-based synthesis (Ollama or OpenAI)  
- FastAPI backend with HDFS support  
- Streamlit frontend for interactive queries

### Features
- **Data Ingestion**  
  - Scrapes PubMed for “SOTA” vs. legacy reviews  
  - Downloads NICE/WHO PDFs, extracts & embeds text  
- **Retrieval**  
  - Spark SQL ranking by keyword overlap  
  - FAISS similarity search over guideline snippets  
  - DuckDuckGo snippet search  
- **Language Models**  
  - LangChain/Ollama integration with templated fallback  
  - ChatOpenAI (gpt-3.5-turbo) fallback if Ollama unavailable  
- **API & UI**  
  - FastAPI endpoints for job submission/status  
  - Asynchronous background jobs with Spark/HDFS  
  - Streamlit app with dataset info, submission form, polling

### Architecture
```mermaid
graph TD
  subgraph UI
    A[Streamlit Frontend]
  end
  subgraph API
    B[FastAPI Backend]
    B --> C[MapReduce (PySpark)]
    B --> D[MongoDB Guidelines]
    B --> E[DuckDuckGo Search]
    C --> F[HDFS / Local JSON]
    D --> G[FAISS Index]
    E --> H[Web Snippets]
    C & G & H --> I[LLM Synthesis]
    I --> B
  end
  A --> B

Getting Started
Prerequisites
Python 3.8+
Java 11+
Hadoop 3.x (HDFS)
Spark 3.x
MongoDB 4.x+
(Optional) Ollama for local LLMs
.env file with API keys & paths
Installation
git clone https://github.com/your-org/FD-Project.git
cd FD-Project
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt

Configuration

Create a file named .env in project root:

MONGO_URI=mongodb://localhost:27017
NCBI_API_KEY=your_pubmed_api_key
HADOOP_HOME=/path/to/hadoop
SPARK_HOME=/path/to/spark
JAVA_HOME=/path/to/java
OLLAMA_HOST=localhost
OLLAMA_PORT=11434

Data Ingestion
Clinical Guidelines
python dataset/data_ingestion/ingest_anxiety_guidelines.py \
  --mongo "$MONGO_URI" \
  --db anxiety --col guidelines \
  --folder dataset/guidelines

Downloads NICE CG159, CG113, WHO mhGAP PDFs
Extracts text via pdfminer or pypdf
Splits into parent (2 000 chars) & child (500 chars) chunks
Computes embeddings with sentence-transformers/all-MiniLM-L6-v2
Upserts into MongoDB with partial unique indexes
PubMed Papers
python dataset/data_ingestion/pbmed_scraper_psy.py \
  --max-sota 500 --max-legacy 350

Queries Entrez ESearch & EFetch
Parses XML abstracts via BeautifulSoup
Outputs dataset/anxiety_papers.json
Tree Generation Utility
python generate_tree.py

Produces:
parent_child_counts.json (child counts per parent chunk)
parent_with_first_child.json (samples of first child embedding)
Usage
CLI Q&A
python anxiety_qa_langchain.py \
  --job_id 12345 \
  --question_file path/to/question.txt \
  --dataset_path dataset/anxiety_papers.json

MapReduce to retrieve top‐10 abstracts
LangChain + Ollama/OpenAI for answer
Fallback template if LLM fails
Writes results/results_<job_id>.json
FastAPI Backend
uvicorn app:app --reload --host 0.0.0.0 --port 8000

POST /submit_question/ → { job_id, status }
GET /job_status/{job_id} → result or status
GET /recent_jobs/ → last 10 jobs
GET /dataset_info → HDFS dataset metadata
Streamlit Frontend
streamlit run streamlit_app.py

Interactive Q&A UI
Sidebar shows HDFS dataset status
Polls backend for job completion
Displays question, answer, timing, citations
Directory Structure
FD-Project/
├── anxiety_qa_langchain.py      # CLI Spark + LangChain QA
├── app.py                       # FastAPI backend
├── dataset/
│   ├── anxiety_papers.json      # PubMed JSON dump
│   └── data_ingestion/
│       ├── guidelines/          # PDF files
│       ├── guidelines_catalogue.json
│       ├── ingest_anxiety_guidelines.py
│       └── pbmed_scraper_psy.py
├── generate_tree.py             # Tree-generation utility
├── my_structure.txt             # Static project tree
├── parent_child_counts.json     # Sample counts output
├── parent_with_first_child.json # Sample first-child output
├── presentation.pptx            # Slides
├── rapport.pdf                  # Written report
├── requirements.txt             # Python deps
├── results/                     # JSON results per job
└── streamlit_app.py             # Streamlit frontend

Data Formats
dataset/anxiety_papers.json
JSON array of PubMed articles:
[
  {
    "pmid": "12345678",
    "title": "...",
    "abstract": "...",
    "journal": "...",
    "year": "2024",
    "authors": ["A. One", "B. Two"]
  },
  …
]

MongoDB guidelines collection
Documents with doc_level: "parent" or "child", plus embeddings for children.
Results files
results/results_<job_id>.json:
{
  "question": "...",
  "answer": "...",
  "error": null,
  "timestamp": "2025-04-30T12:34:56"
}

Dependencies

See requirements.txt for:

fastapi
uvicorn
streamlit
pyspark
hadoop-client
pymongo
sentence-transformers
langchain
langchain-ollama
beautifulsoup4
pdfminer.six
pypdf
faiss-cpu
duckduckgo-search
python-dotenv

Contributing
Fork & clone
Create feature branch (git checkout -b feat/xyz)
Install & run tests (if any)
Submit PR with clear description & tests
License

This project is licensed under the MIT License. See LICENSE.

Author & Contact

Your Name
GitHub: @BSHLoussama
Email: oussamaboussahla2017@gmail.com
