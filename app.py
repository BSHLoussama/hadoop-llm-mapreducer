# app.py - FastAPI backend with HDFS integration and Python path fixes
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import subprocess
import os
import json
import tempfile
import sys
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Biomedical MapReduce QA System")

class Question(BaseModel):
    text: str
    dataset_path: str = "/user/oussama/biomedical_abstracts.json"  # HDFS path

class Answer(BaseModel):
    question: str
    answer: str
    processing_time: float
    date: str

# Store recent job results
job_results: Dict[str, Dict[str, Any]] = {}

@app.post("/submit_question/", response_model=Dict[str, str])
async def submit_question(question: Question, background_tasks: BackgroundTasks):
    """Submit a question for processing with Hadoop/PySpark"""
    job_id = f"job_{datetime.now().strftime('%Y%m%d%H%M%S')}"

    logger.info(f"Submitting job {job_id} with question: {question.text}")
    logger.info(f"Dataset path: {question.dataset_path}")

    # Schedule the job to run in the background
    background_tasks.add_task(
        run_mapreduce_job,
        job_id=job_id,
        question=question.text,
        dataset_path=question.dataset_path
    )

    return {"job_id": job_id, "status": "submitted"}

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
        script_path = os.path.join(current_dir, "bio_qa_langchain.py")

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
        result_path = f"results_{job_id}.json"
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
                "answer": "This is a simulated answer about genetic mutations and their impact on cellular functions. The mutations affect signaling pathways and protein interactions.",
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
        cmd = [hdfs_cmd, "dfs", "-ls", "/user/oussama/biomedical_abstracts.json"]
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
                "location": "/user/oussama/biomedical_abstracts.json",
                "source": "HDFS"
            }
        else:
            return {
                "exists": False,
                "error": stderr if stderr else "File not found in HDFS",
                "format": "Unknown",
                "location": "/user/oussama/biomedical_abstracts.json",
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