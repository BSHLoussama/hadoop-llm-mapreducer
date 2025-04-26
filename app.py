# app.py - FastAPI backend with HDFS integration and Python path fixes
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import subprocess
import time
import json
import tempfile
from datetime import datetime
from functools import lru_cache
import logging
import os
import sys


if hasattr(sys.stdout, "reconfigure"):
    # Python 3.7+ on Windows
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
else:
    # fallback for older versions – set the PYTHONUTF8 env var
    os.environ["PYTHONUTF8"] = "1"


# Optional LangChain / Ollama imports
try:
    from langchain_ollama import OllamaLLM
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import PromptTemplate
    from langchain_core.runnables import RunnableLambda
    from langchain_huggingface import HuggingFaceEmbeddings
    from pymongo import MongoClient
    from langchain_community.vectorstores import FAISS
    from langchain_core.documents import Document
    from langchain_community.tools import DuckDuckGoSearchRun
    from dotenv import load_dotenv

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

# create and check the dir results
# Define the directory name
results_dir = "results"

# Check and create if it doesn't exist
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
    print(f"Created directory: {results_dir}")
else:
    print(f"Directory already exists: {results_dir}")
    
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("anxiety_qa")

app = FastAPI(title="Anxiety QA Platform")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# Load environment variables from .env
load_dotenv()
# ── Data Models ───────────────────────────────────────────────────────────
class Question(BaseModel):
    text: str
    dataset_path: str = "/user/oussama/anxiety_papers.json"  # HDFS path

class Answer(BaseModel):
    question: str
    answer: str
    processing_time: float
    date: str

class JobResult(BaseModel):
    question: str
    answer: str
    source: str
    status: str
    processing_time: float
    date: str
    
# Store recent job results
job_results: Dict[str, Dict[str, Any]] = {}

# ── LLM & Synthesiser ─────────────────────────────────────────────────────
if not LANGCHAIN_AVAILABLE:
    logger.warning("LangChain/Ollama unavailable: LLM features disabled.")

# LLM = OllamaLLM(model="llama3.2:1b", temperature=0.0) if LANGCHAIN_AVAILABLE else None
LLM = ChatOpenAI(model="gpt-3.5-turbo")  if LANGCHAIN_AVAILABLE else None

_SYNTH_PROMPT = PromptTemplate(
    input_variables=["partials"],
    template=(
        "You are a medical synthesis expert. Combine these partial answers "
        "into one coherent, structured reply.\n\n"
        "Rules:\n"
        "1. Use ONLY the provided content; do NOT hallucinate or add new information.\n"
        "2. If a part of the question isn’t covered, state exactly:\n"
        "   \"I don't know based on current data.\"\n"
        "3. Cite each fact with its source tag in square brackets: [PAPERS], [GUIDELINES], [NET].\n"
        "4. Think step-by-step:\n"
        "   a. Identify key findings in each partial.\n"
        "   b. Organize findings into logical sections with headers.\n"
        "   c. Draft the final answer, embedding citations.\n\n"
        "Partial answers:\n"
        "{partials}\n\n"
        "Begin your step-by-step reasoning:"
    )
)

# Create the synthesis chain using LCEL if LangChain is available
synth_chain = (_SYNTH_PROMPT | LLM) if LANGCHAIN_AVAILABLE else None


# ── NET via DuckDuckGo ───────────────────────────────────────────────────
net_tool = DuckDuckGoSearchRun() if LANGCHAIN_AVAILABLE else None

_NET_PROMPT = PromptTemplate(
    input_variables=["question", "snips"],
    template=(
        "You are an expert researcher. Use ONLY the following web snippets "
        "to answer the question. Do NOT hallucinate or add information not in the snippets.\n\n"
        "Rules:\n"
        "1. If the snippets don’t answer the question, respond with:\n"
        "   \"I don't know based on the provided snippets.\"\n"
        "2. Cite each statement with the snippet number in brackets, e.g. [1], [2].\n"
        "3. Structure your output in three parts:\n"
        "   1) Step-by-step identification of relevant info\n"
        "   2) A short summary of findings\n"
        "   3) Final concise answer with citations\n\n"
        "Question: {question}\n\n"
        "Web Snippets:\n"
        "{snips}\n\n"
        "Begin your reasoning and answer:"
    )
)

# LCEL-based chain
net_chain = (_NET_PROMPT | LLM) if LANGCHAIN_AVAILABLE else None

def process_net(q: str) -> str:
    if not LANGCHAIN_AVAILABLE:
        return "Web search unavailable."
    
    try:
        res = net_tool.run(q + " anxiety")
    except Exception as e:
        logger.warning("Web search failed: %s", e)
        return "Web search unavailable."

    # Format the snippets
    snips = res if isinstance(res, str) else "\n".join(
        f"{r.get('title', '')} – {r.get('snippet', '')}" for r in res[:8]
    )

    # Run the LLM pipeline
    try:
        return net_chain.invoke({"question": q, "snips": snips}).content
    except Exception as e:
        logger.error("Net chain error: %s", e)
        return "Failed to generate web-based answer."
# __ MapReduce+LLMs ________________________________________________________

def run_mapreduce_job(job_id: str, question: str, dataset_path: str):
    """Run the PySpark job to process data from HDFS"""
    # Save question to a temporary file
    question_file = os.path.join(tempfile.gettempdir(), f"question_{job_id}.txt")
    with open(question_file, "w", encoding="utf-8") as f:
        f.write(question)

    logger.info(f"Question saved to temporary file: {question_file}")

    try:
        # Start timing
        start_time = datetime.now()

        # Environment setup for Spark and Hadoop
        env = os.environ.copy()
        hadoop_home = os.environ.get("HADOOP_HOME", "C:\\hadoop-3.3.6")
        spark_home = os.environ.get("SPARK_HOME", "C:\\spark-3.3.2-bin-hadoop3")

        # Get the full path to Python executable (THIS IS THE KEY FIX)
        python_exe = sys.executable  # Use the same Python that's running this app

        # Get the current working directory for script path resolution
        current_dir = os.getcwd()
        script_path = os.path.join(current_dir, "anxiety_qa_langchain.py")

        # Ensure Hadoop and Spark binaries are in the PATH
        if "bin" not in env.get("PATH", ""):
            env["PATH"] = os.path.join(hadoop_home, "bin") + os.pathsep + env.get("PATH", "")
            env["PATH"] = os.path.join(spark_home, "bin") + os.pathsep + env.get("PATH", "")

        # Ensure JAVA_HOME is set
        if "JAVA_HOME" not in env:
            env["JAVA_HOME"] = os.environ.get("JAVA_HOME", "C:\\Program Files\\Java\\jdk-11")

        # Log paths for debugging
        logger.info(f"HADOOP_HOME: {hadoop_home}")
        logger.info(f"SPARK_HOME: {spark_home}")
        logger.info(f"JAVA_HOME: {env.get('JAVA_HOME')}")
        logger.info(f"Using Python at: {python_exe}")
        logger.info(f"Script path: {script_path}")

        # TWO OPTIONS: SPARK APPROACH OR DIRECT PYTHON APPROACH

        # OPTION 1: Try running with spark-submit (with Python path fix)
        if True:  # Change to False to use Option 2
            # Build the spark-submit command with explicit Python path
            cmd = [
                os.path.join(spark_home, "bin", "spark-submit.cmd"),
                "--master", "local[*]",  # Run in local mode with as many threads as cores
                "--conf", "spark.hadoop.fs.defaultFS=hdfs://localhost:9000",  # HDFS URI
                "--conf", "spark.driver.memory=2g",  # Allocate memory for driver
                "--conf", "spark.executor.memory=1g",  # Allocate memory for executor
                "--conf", f"spark.pyspark.python={python_exe}",  # Tell Spark which Python to use
                script_path,  # Full path to Python script
                "--job_id", job_id,
                "--question_file", question_file,
                "--dataset_path", dataset_path
            ]
        else:
            # OPTION 2: Directly call Python script (no Spark, simpler for testing)
            cmd = [
                python_exe,
                script_path,
                "--job_id", job_id,
                "--question_file", question_file,
                "--dataset_path", dataset_path
            ]

        # Log the full command for debugging
        logger.info(f"Executing command: {' '.join(cmd)}")

        # Run the process with appropriate environment
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            text=True  # Return stdout/stderr as strings
        )

        # Capture output
        stdout, stderr = process.communicate()

        # Log output for debugging
        logger.info(f"Process stdout: {stdout}")
        if stderr:
            logger.warning(f"Process stderr: {stderr}")

        # Log return code
        logger.info(f"Process return code: {process.returncode}")

        # End timing
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()

        # Check for results file
        result_path = f"results/results_{job_id}.json"
        if os.path.exists(result_path):
            logger.info(f"Results file found: {result_path}")

            # Read results
            with open(result_path, "r", encoding="utf-8") as f:
                result = json.load(f)

            # Store in memory
            job_results[job_id] = {
                "question": question,
                "answer": result.get("answer", "No answer generated"),
                "processing_time": processing_time,
                "date": start_time.strftime("%Y-%m-%d %H:%M:%S"),
                "status": "completed"
            }

            logger.info(f"Job {job_id} completed successfully")
        else:
            logger.warning(f"Results file not found: {result_path}")

            # Fallback to simulated response
            job_results[job_id] = {
                "question": question,
                "answer": "This is a simulated answer about anxiety and its impact on mental and physical health. Anxiety can affect brain chemistry, influence thought patterns, and trigger physical responses such as increased heart rate and muscle tension.",
                "processing_time": processing_time,
                "date": start_time.strftime("%Y-%m-%d %H:%M:%S"),
                "status": "simulated"  # Indicate this is a fallback
            }

            logger.warning(f"Generated simulated response as results file was not found.")
            print(f"Note: Generated simulated response as results file was not found.")

    except Exception as e:
        logger.error(f"Job failed with error: {str(e)}", exc_info=True)

        # Record failure
        job_results[job_id] = {
            "status": "failed",
            "error": str(e),
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        print(f"Job failed: {str(e)}")

def process_papers(job_id: str, question: str, dataset_path: str) -> str:
    """
    Synchronously run the existing run_mapreduce_job (which will
    populate job_results[job_id]) and then pull back the answer.
    """
    # Call your existing orchestration in-process
    run_mapreduce_job(job_id, question, dataset_path)

    # Now the job_results dict must contain an entry for job_id
    result = job_results.get(job_id, {})
    # Return the answer (or a fallback if something went wrong)
    return result.get("answer", "No answer generated")

# ── Routing ───────────────────────────────────────────────────────────────
_ROUTER_PROMPT = PromptTemplate(
    input_variables=["question"],
    template=(
        "You are a routing assistant. Based on the user's question, choose the most "
        "appropriate sources from: PAPERS, GUIDELINES, NET.\n\n"
        "Rules:\n"
        "1. Include PAPERS for research-based evidence.\n"
        "2. Include GUIDELINES for official recommendations.\n"
        "3. Include NET for the latest web-based context.\n"
        "4. Always choose at least one; if unsure, default to PAPERS.\n\n"
        "Question: {question}\n"
        "Answer with a comma-separated list of choices (uppercase), no extra text."
    )
)

router_chain = (_ROUTER_PROMPT | LLM) if LANGCHAIN_AVAILABLE else None

def choose_sources(question: str) -> List[str]:
    if LANGCHAIN_AVAILABLE and router_chain is not None:
        try:
            out = router_chain.invoke({"question": question})
            picks = [x.strip().lower() for x in out.content.split(",")]
            valid = [x for x in picks if x in {"papers", "guidelines", "net"}]
            if valid:
                return valid
        except Exception as e:
            logger.warning("Router chain failed: %s", e)

    # Heuristic fallback
    low = question.lower()
    heur = set()
    if "guideline" in low or "recommend" in low:
        heur.add("guidelines")
    if "internet" in low or "website" in low or "online" in low:
        heur.add("net")
    return list(heur or {"papers"})


# ── MongoDB & Guidelines (Vector Only) ────────────────────────────────────
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
mongo_cli = MongoClient(MONGO_URI)
mongo_col = mongo_cli["anxiety"]["guidelines"]
EMBED_MODEL = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
) if LANGCHAIN_AVAILABLE else None

@lru_cache()
def get_guidelines_index() -> FAISS:
    docs: List[Document] = []
    for d in mongo_col.find(
        {"doc_level": "child"}, {"text":1, "parent_id":1, "title":1, "_id":0}
    ):
        docs.append(Document(
            page_content=d["text"],
            metadata={"parent_id": d["parent_id"], "title": d["title"]}
        ))
    return FAISS.from_documents(docs, EMBED_MODEL)

def process_guidelines(q: str) -> str:
    """Top-5 FAISS vector search over guideline snippets."""
    if not LANGCHAIN_AVAILABLE:
        return "Guidelines processing unavailable."
    idx = get_guidelines_index()
    hits = idx.similarity_search(q, k=5)
    if not hits:
        return "No relevant guidelines found."
    snippets = []
    for doc in hits:
        title = doc.metadata.get("title", "Untitled")
        txt   = doc.page_content[:1000].strip().replace("\n"," ")
        snippets.append(f"## {title}\n{txt}…")
    return "\n\n".join(snippets)



# ── Worker & Endpoints ────────────────────────────────────────────────────
def worker(job_id: str, question: str, data_path: str):
    t0 = time.time()
    parts: List[str] = []

    for src in choose_sources(question):
        if src == "guidelines":
            parts.append("[GUIDELINES]\n" + process_guidelines(question))
            
        elif src == "net":
            parts.append("[NET]\n" + process_net(question))
        else:  # "papers"
            # This will synchronously run spark/HDFS and populate job_results[job_id]
            ans = process_papers(job_id, question, data_path)
            parts.append("[PAPERS]\n" + ans)

    # Combine or pass through
    if len(parts) == 1:
        final = parts[0].split("]\n", 1)[1]
    else:
        combined = "\n\n".join(parts)
        final = (
            synth_chain.invoke({"partials": combined}).content
            if LANGCHAIN_AVAILABLE
            else combined
        )

    # Overwrite job_results[job_id] with our final synthesized result
    job_results[job_id] = JobResult(
        question=question,
        answer=final,
        source=",".join(choose_sources(question)),
        status="completed",
        processing_time=time.time() - t0,
        date=datetime.utcnow().isoformat()
    ).model_dump()

    logger.info("Job %s done in %.2f s", job_id, time.time() - t0)
    
    
@app.post("/submit_question/", response_model=Dict[str, str])
async def submit_question(q: Question, background_tasks: BackgroundTasks):
    job_id = f"job_{int(time.time() * 1000)}"
    # mark it as running immediately
    job_results[job_id] = {"status": "running", "question": q.text}
    # schedule our unified worker
    background_tasks.add_task(worker, job_id, q.text, q.dataset_path)
    return {"job_id": job_id, "status": "submitted"}


@app.get("/job_status/{job_id}")
async def get_job_status(job_id: str):
    """Get the status and results of a submitted job"""
    if job_id not in job_results:
        return {"status": "not_found"}

    return job_results[job_id]

@app.get("/recent_jobs/")
async def get_recent_jobs():
    """Get a list of recent jobs and their statuses"""
    # Return the 10 most recent jobs (if they exist)
    job_ids = list(job_results.keys())
    recent_job_ids = job_ids[-10:] if len(job_ids) > 10 else job_ids

    return {job_id: {
        "status": job_results[job_id].get("status", "unknown"),
        "date": job_results[job_id].get("date", "unknown")
    } for job_id in recent_job_ids}

@app.get("/dataset_info")
async def get_dataset_info():
    """Get information about the HDFS dataset"""
    try:
        # Try to get dataset info from HDFS using hadoop command
        hadoop_home = os.environ.get("HADOOP_HOME", "C:\\hadoop-3.3.6")
        hdfs_cmd = os.path.join(hadoop_home, "bin", "hdfs.cmd")

        # Check if file exists
        cmd = [hdfs_cmd, "dfs", "-ls", "/user/oussama/anxiety_papers.json"]
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        stdout, stderr = process.communicate()

        if process.returncode == 0 and stdout.strip():
            # File exists, get file details
            file_details = stdout.strip().split("\n")[-1].split()
            file_size = int(file_details[4]) if len(file_details) > 4 else 0

            # Get file modification time
            last_modified = " ".join(file_details[5:8]) if len(file_details) > 7 else "Unknown"

            return {
                "exists": True,
                "size_bytes": file_size,
                "size_formatted": f"{file_size / 1024:.2f} KB" if file_size < 1024*1024 else f"{file_size / (1024*1024):.2f} MB",
                "last_modified": last_modified,
                "format": "JSON",
                "location": "/user/oussama/anxiety_papers.json",
                "source": "HDFS"
            }
        else:
            return {
                "exists": False,
                "error": stderr if stderr else "File not found in HDFS",
                "format": "Unknown",
                "location": "/user/oussama/anxiety_papers.json",
                "note": "Dataset might not be uploaded to HDFS yet"
            }
    except Exception as e:
        logger.error(f"Error getting dataset info: {str(e)}", exc_info=True)
        return {
            "exists": "unknown",
            "error": str(e),
            "note": "Could not determine dataset status"
        }

@app.get("/")
async def root():
    """Root endpoint for quick status check"""
    return {
        "status": "online",
        "service": "Biomedical MapReduce QA System",
        "endpoints": {
            "submit_question": "POST /submit_question/",
            "job_status": "GET /job_status/{job_id}",
            "recent_jobs": "GET /recent_jobs/",
            "dataset_info": "GET /dataset_info"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)