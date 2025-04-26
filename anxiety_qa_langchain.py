#!/usr/bin/env python3
"""
anxiety_qa.py

Anxiety Question Answering with MapReduce (Spark) and LangChain/Ollama.
Processes anxiety-related abstracts and answers user questions using:
  1. A Spark SQL MapReduce step to retrieve relevant abstracts.
  2. An LLM (llama3.2:1b via Ollama) through LangChain, with a
     template‐based fallback if the LLM is unavailable or errors out.
Outputs results to a JSON file named results_<job_id>.json.
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional

# Load environment variables from .env
load_dotenv()

if hasattr(sys.stdout, "reconfigure"):
    # Python 3.7+ on Windows
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
else:
    # fallback for older versions – set the PYTHONUTF8 env var
    os.environ["PYTHONUTF8"] = "1"

# -----------------------------------------------------------------------------
# Logging Configuration
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("AnxietyQA")

# -----------------------------------------------------------------------------
# Optional LangChain/Ollama Imports
# -----------------------------------------------------------------------------
try:
    from langchain_ollama import OllamaLLM
    from langchain_openai import ChatOpenAI
    from langchain.prompts import PromptTemplate
    from langchain_core.runnables import RunnableLambda
    LANGCHAIN_AVAILABLE = True
    logger.info("LangChain and langchain_ollama detected: LLM support enabled.")
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logger.warning("LangChain or langchain_ollama not available. Using template fallback.")

# -----------------------------------------------------------------------------
# Argument Parsing
# -----------------------------------------------------------------------------
def parse_arguments() -> argparse.Namespace:
    """Parse command‐line arguments."""
    parser = argparse.ArgumentParser(
        description="Anxiety QA with MapReduce and LangChain/Ollama"
    )
    parser.add_argument("--job_id",       required=True, help="Unique job identifier")
    parser.add_argument("--question_file",required=True, help="Path to text file with the question")
    parser.add_argument("--dataset_path", required=True, help="Path to JSON dataset of abstracts")
    return parser.parse_args()

# -----------------------------------------------------------------------------
# Read User Question
# -----------------------------------------------------------------------------
def read_question(file_path: str) -> str:
    """Read the question from a file or return a default."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            q = f.read().strip()
            if q:
                return q
    except Exception as e:
        logger.error("Failed to read question file '%s': %s", file_path, e)

    # Default fallback question
    default_q = "What are effective treatments for generalized anxiety disorder?"
    logger.info("Using default question: %s", default_q)
    return default_q

# -----------------------------------------------------------------------------
# MapReduce Step: Retrieve Relevant Abstracts via Spark SQL
# -----------------------------------------------------------------------------
def find_relevant_abstracts(df, question: str, spark) -> List[Dict[str, Any]]:
    """
    Use Spark SQL to rank abstracts by simple keyword overlap.
    Returns up to 10 top‐scoring records as dicts.
    """
    df.createOrReplaceTempView("abstracts")

    # Define a small set of stopwords to ignore
    stopwords = {
        "what","where","when","which","how","are","the","and","for","with","that",
        "these","those","from","this","is","to","in","of","an","a"
    }
    # Extract candidate keywords (>3 chars, not stopwords)
    keywords = [
        w.lower().strip(".,?()")
        for w in question.split()
        if len(w) > 3 and w.lower() not in stopwords
    ]
    if not keywords:
        keywords = ["anxiety"]  # ensure at least one term

    # Build SQL WHERE clause and relevance score expression
    conditions = [
        f"lower(abstract) LIKE '%{kw}%'" for kw in keywords
    ]
    where_clause = " OR ".join(conditions)
    score_expr = " + ".join(
        f"(CASE WHEN lower(abstract) LIKE '%{kw}%' THEN 1 ELSE 0 END)"
        for kw in keywords
    )

    query = f"""
        SELECT
            title,
            abstract,
            {score_expr} AS relevance_score
        FROM abstracts
        WHERE {where_clause}
        ORDER BY relevance_score DESC
        LIMIT 10
    """

    try:
        results = spark.sql(query).collect()
        abstracts = [row.asDict() for row in results]
        logger.info("Retrieved %d relevant abstracts", len(abstracts))
        return abstracts
    except Exception as e:
        logger.error("Error during Spark SQL retrieval: %s", e)
        return []

# -----------------------------------------------------------------------------
# Generate Answer: LLM + Fallback
# -----------------------------------------------------------------------------
def generate_answer(question: str, abstracts: List[Dict[str, Any]]) -> str:
    """
    Generate the answer string using:
      1. LLMChain + OllamaLLM if available, else
      2. Simple template‐based summarization.
    """
    if not abstracts:
        return "I couldn't find relevant information on anxiety in the dataset."

    # Prepare context: top 5 abstracts
    context = "\n\n".join(
        f"Title: {a.get('title','')} \nAbstract: {a.get('abstract','')}"
        for a in abstracts[:5]
    )

    # Attempt LLM generation
    if LANGCHAIN_AVAILABLE:
        try:
            template=(
                "You are an expert anxiety research assistant. Use ONLY the provided abstracts "
                "to answer the question. Do NOT hallucinate or add external content.\n\n"
                "Rules:\n"
                "1. If an abstract doesn’t cover a point, say:\n"
                "   \"I don't know based on provided abstracts.\"\n"
                "2. Structure your response:\n"
                "   1) Step-by-step reasoning: identify which abstracts are relevant\n"
                "   2) Bullet-point key findings, each tagged with the abstract title in brackets\n"
                "   3) A final concise answer paragraph\n\n"
                "Question: {question}\n\n"
                "Abstracts (top 5):\n"
                "{abstracts}\n\n"
                "Begin your reasoning:"
            )
            prompt = PromptTemplate(
                input_variables=["question", "abstracts"],
                template=template
            )
            # llm = OllamaLLM(model="llama3.2:1b", temperature=0.1)
            # Then you can just do
            llm = ChatOpenAI(model="gpt-3.5-turbo") 
            chain = prompt | llm
            response = chain.invoke({"question": question, "abstracts": context})
            logger.info("Answer generated via LLM.")
            return response.content.strip()
        except Exception as e:
            logger.error("LLM generation failed: %s", e)

    # Fallback: template‐based summary
    logger.info("Using template-based fallback.")
    ql = question.lower()
    if "treatment" in ql or "therapy" in ql:
        points = [
            "Cognitive Behavioral Therapy (CBT) shows consistent efficacy",
            "Selective Serotonin Reuptake Inhibitors (SSRIs) are first-line medications",
            "Lifestyle changes (exercise, sleep hygiene) can reduce symptoms"
        ]
    elif "symptom" in ql:
        points = [
            "Excessive worry and restlessness are core symptoms",
            "Muscle tension and sleep disturbances frequently co-occur",
            "Concentration difficulties are common"
        ]
    else:
        points = [
            "Anxiety involves hyperactivation of the amygdala",
            "Genetic and environmental factors both play roles",
            "Early intervention improves long-term outcomes"
        ]

    # Craft fallback answer
    answer = (
        f"Based on the literature: {points[0]}. {points[1]}."
        f" {points[2]}."
    )
    return answer

# -----------------------------------------------------------------------------
# Save Results to JSON
# -----------------------------------------------------------------------------
def save_results(job_id: str, question: str, answer: str, error: Optional[str] = None):
    """Write out question, answer, and optional error to results_<job_id>.json"""
    payload = {
        "question": question,
        "answer": answer,
        "error": error,
        "timestamp": datetime.now().isoformat()
    }
    filename = f"results/results_{job_id}.json"
    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        logger.info("Results saved to %s", filename)
    except Exception as e:
        logger.error("Failed to save results: %s", e)

# -----------------------------------------------------------------------------
# Main Workflow
# -----------------------------------------------------------------------------
def main():
    args = parse_arguments()
    question = read_question(args.question_file)
    logger.info("Question: %s", question)
    logger.info("Dataset: %s", args.dataset_path)

    try:
        from pyspark.sql import SparkSession

        # Initialize Spark
        spark = SparkSession.builder \
            .appName(f"AnxietyQA-{args.job_id}") \
            .getOrCreate()

        # Read JSON dataset (one JSON object per line or multiline)
        df = spark.read.option("multiline", "true").json(args.dataset_path)
        cols = set(df.columns)
        if not {"title", "abstract"}.issubset(cols):
            err = f"Dataset missing required columns. Found: {cols}"
            logger.error(err)
            save_results(args.job_id, question, "", error=err)
            return

        # Retrieve and answer
        abstracts = find_relevant_abstracts(df, question, spark)
        answer = generate_answer(question, abstracts)
        save_results(args.job_id, question, answer)

    except Exception as e:
        err_msg = f"Processing error: {e}"
        logger.exception(err_msg)
        save_results(args.job_id, question, "", error=err_msg)

    finally:
        # Clean up Spark
        if 'spark' in locals():
            spark.stop()
            logger.info("Spark session stopped.")

if __name__ == "__main__":
    main()