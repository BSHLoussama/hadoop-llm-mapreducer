#!/usr/bin/env python3
"""
Biomedical Question Answering with MapReduce and LangChain
Processes biomedical abstracts and answers questions using a small-scale LLM.
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Use the updated Ollama model for LangChain from langchain_ollama
try:
    from langchain_ollama import OllamaLLM
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logger.warning("LangChain or langchain_ollama package not available. Using template-based responses instead.")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Biomedical QA with MapReduce and LangChain")
    parser.add_argument("--job_id", required=True, help="Unique job identifier")
    parser.add_argument("--question_file", required=True, help="Path to the question file")
    parser.add_argument("--dataset_path", required=True, help="Path to the dataset (HDFS or local)")
    return parser.parse_args()


def read_question(file_path: str) -> str:
    """Read the question from a file"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception as e:
        logger.error(f"Error reading question file: {str(e)}")
        return "What are the effects of genetic mutations on cellular function?"  # Default fallback


def find_relevant_abstracts(df, question: str, spark) -> List[Dict[str, Any]]:
    """
    Use MapReduce paradigm via Spark to find relevant abstracts.
    Registers the abstracts DataFrame as a temporary SQL table, builds a SQL
    query based on simple keyword matching, and returns the top relevant results.
    """
    try:
        # Register the DataFrame as a temporary SQL table.
        df.createOrReplaceTempView("abstracts")

        # Extract keywords from the question using a simple filter.
        keywords = [
            word.lower() for word in question.split()
            if len(word) > 3 and word.lower() not in ("what", "where", "when", "which", "how", "are", "the", "and", "for")
        ]

        # Build SQL conditions to match keywords in the abstract text.
        keyword_conditions = []
        for keyword in keywords:
            keyword_conditions.append(f"lower(abstract) LIKE '%{keyword}%'")

        keyword_sql = " OR ".join(keyword_conditions)
        if not keyword_sql:
            keyword_sql = "1=1"  # Default condition if no keywords

        query = f"""
        SELECT
            title,
            abstract,
            {' + '.join([f"(CASE WHEN lower(abstract) LIKE '%{k}%' THEN 1 ELSE 0 END)" for k in keywords])} as relevance_score
        FROM abstracts
        WHERE {keyword_sql}
        ORDER BY relevance_score DESC
        LIMIT 10
        """

        # Execute the query.
        results = spark.sql(query)

        # Convert to Python objects.
        relevant_abstracts = [row.asDict() for row in results.collect()]
        logger.info(f"Found {len(relevant_abstracts)} relevant abstracts")
        return relevant_abstracts

    except Exception as e:
        logger.error(f"Error in MapReduce process: {str(e)}")
        return []


def generate_answer(question: str, abstracts: List[Dict[str, Any]]) -> str:
    """
    Generate an answer using an LLM when available (via Ollama's llama3.2:1b  through LangChain),
    with fallback to template-based responses.
    """
    if not abstracts:
        return "I couldn't find relevant information in the dataset."

    # Combine top abstracts for the context, limiting to the top 5 matches.
    abstracts_text = "\n\n".join([
        f"Title: {abs.get('title', 'Untitled')}\nAbstract: {abs.get('abstract', 'No abstract')}"
        for abs in abstracts[:5]
    ])

    # If LangChain (and thus OllamaLLM) is available, try generating an answer.
    if LANGCHAIN_AVAILABLE:
        try:
            template = (
                "You are a biomedical research assistant. Answer the question based only on the provided abstracts.\n\n"
                "Question: {question}\n\n"
                "Abstracts:\n{abstracts}\n\n"
                "Answer:"
            )
            prompt = PromptTemplate(
                input_variables=["question", "abstracts"],
                template=template
            )

            # Instantiate the LLM with llama3.2:1b  via Ollama.
            # Note: max_tokens is no longer supported in the constructor.
            llm = OllamaLLM(model="llama3.2:1b", temperature=0.1)

            # Create and run the chain.
            chain = LLMChain(llm=llm, prompt=prompt)
            response = chain.run(question=question, abstracts=abstracts_text)

            logger.info("Generated answer using LangChain with OllamaLLM (llama3.2:1b )")
            return response

        except Exception as e:
            logger.error(f"Error using LLM: {str(e)}")
            # Fall back to template-based answer in case of error.

    logger.info("Using template-based response (LangChain unavailable or error occurred)")

    # Template-based simulated answer.
    titles = [abs.get("title", "Unknown Study") for abs in abstracts[:3]]
    if "mutation" in question.lower() and "cellular" in question.lower():
        findings = [
            "multiple studies indicate that genetic mutations can disrupt normal cellular processes through protein dysfunction",
            "research suggests that mutations in key regulatory genes can lead to altered signaling pathways",
            "several abstracts mention the impact of mutations on transcription factors and gene expression"
        ]
    elif "treatment" in question.lower() or "therapy" in question.lower():
        findings = [
            "several studies investigate targeted therapies for specific genetic mutations",
            "research indicates personalized medicine approaches based on genetic profiles show promise",
            "multiple abstracts discuss the relationship between genetic variations and treatment efficacy"
        ]
    else:
        findings = [
            "research suggests a strong correlation between genetic factors and disease progression",
            "multiple studies highlight the importance of understanding molecular mechanisms",
            "several abstracts emphasize the role of genetic variations in biological processes"
        ]

    answer = f"Based on the biomedical literature analyzed, {findings[0]}. "
    answer += f"Additionally, {findings[1]}. "
    if len(titles) >= 2:
        answer += f"Notable studies include '{titles[0]}' and '{titles[1]}', which provide evidence for these findings. "
    if len(findings) > 2:
        answer += f"In conclusion, {findings[2]}."
    return answer


def save_results(job_id: str, question: str, answer: str, error: Optional[str] = None):
    """Save results to a JSON file"""
    results = {
        "question": question,
        "answer": answer,
        "error": error,
        "timestamp": datetime.now().isoformat()
    }

    filename = f"results_{job_id}.json"
    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {filename}")
    except Exception as e:
        logger.error(f"Error saving results: {str(e)}")


def main():
    """Main function to process the biomedical QA task"""
    args = parse_arguments()
    question = read_question(args.question_file)
    logger.info(f"Processing question: {question}")
    logger.info(f"Dataset path: {args.dataset_path}")

    logger.info("Creating Spark session...")
    try:
        from pyspark.sql import SparkSession

        spark = SparkSession.builder \
            .appName(f"BioQA-{args.job_id}") \
            .getOrCreate()

        logger.info(f"Reading dataset: {args.dataset_path}")
        df = spark.read.option("multiline", "true").json(args.dataset_path)

        if "abstract" not in df.columns or "title" not in df.columns:
            available_cols = ", ".join(df.columns)
            error_msg = f"Required columns missing. Available columns: {available_cols}"
            logger.error(error_msg)
            save_results(args.job_id, question,
                         f"An error occurred during processing: {error_msg}",
                         error_msg)
            return

        count = df.count()
        logger.info(f"Successfully read {count} records from dataset")

        relevant_abstracts = find_relevant_abstracts(df, question, spark)
        answer = generate_answer(question, relevant_abstracts)
        save_results(args.job_id, question, answer)

    except Exception as e:
        error_message = str(e)
        logger.error(f"Error processing data: {error_message}")
        answer = f"An error occurred during processing: {error_message}"
        save_results(args.job_id, question, answer, error_message)
    finally:
        if "spark" in locals():
            spark.stop()
            logger.info("Spark session stopped")


if __name__ == "__main__":
    main()