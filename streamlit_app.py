import streamlit as st
import requests
import time
from datetime import datetime

# API endpoint - change if needed
API_ENDPOINT = "http://localhost:8000"

# Page configuration with custom theme
st.set_page_config(
    page_title="Anxiety QA Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better appearance
st.markdown("""
<style>
  .main-header {
      font-size: 2.5rem;
      color: #4257b2;
      margin-bottom: 1.5rem;
  }
  .section-header {
      font-size: 1.8rem;
      color: #31333F;
      margin-top: 2rem;
      margin-bottom: 1rem;
  }
  .card-container {
      background-color: #f8f9fa;
      border-radius: 0.5rem;
      padding: 1.5rem;
      margin: 1rem 0;
      border-left: 4px solid #4257b2;
  }
  .success-container {
      background-color: #f0fff4;
      border-left: 4px solid #38a169;
  }
  .error-container {
      background-color: #fff5f5;
      border-left: 4px solid #e53e3e;
  }
  .text-muted {
      color: #718096;
      font-size: 0.875rem;
  }
  .result-heading {
      font-size: 1.2rem;
      font-weight: 600;
      color: #2d3748;
      margin-bottom: 0.5rem;
  }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables if needed
if "job_id" not in st.session_state:
    st.session_state.job_id = None

# Sidebar for dataset info and status remains the same
with st.sidebar:
    st.markdown("### 😰 Anxiety QA System")
    st.markdown("Ask questions about anxiety disorders and treatments.")
    st.markdown("---")
    st.markdown("### Dataset Information")
    try:
        # Calling the dataset_info endpoint on the backend
        response = requests.get(f"{API_ENDPOINT}/dataset_info")
        if response.status_code == 200:
            dataset_info = response.json()
            if dataset_info.get("exists", False):
                st.markdown(f"📊 **File Size:** {dataset_info.get('size_formatted', 'Unknown')}")
                st.markdown(f"⏰ **Last Modified:** {dataset_info.get('last_modified', 'Unknown')}")
                st.markdown("✅ **Status:** Available")
            else:
                st.markdown("❌ **Status:** Not found")
        else:
            st.markdown("⚠️ **Status:** Unable to check")
    except Exception:
        st.markdown("⚠️ **Status:** Connection error")

# Create two tabs: one for asking questions and one for About info
tab1, tab2 = st.tabs(["💬 Ask Questions", "ℹ️ About"])

with tab1:
    st.markdown('<h1 class="main-header">Anxiety QA Platform</h1>', unsafe_allow_html=True)
    st.markdown("""
    Submit your question about anxiety disorders, treatments, and guidelines.
    Our system will process clinical guidelines, research papers, and web results
    to generate a comprehensive answer.
    """)

    # Question submission form in a styled card
    st.markdown('<div class="card-container">', unsafe_allow_html=True)
    default_question = (
        "What are the recommended first‑line treatments for generalized anxiety disorder?"
    )
    question = st.text_area("Your Question:", value=default_question, height=150, key="question_input")

    # Advanced options (e.g., specifying the HDFS dataset path)
    with st.expander("Advanced Options"):
        dataset_path = st.text_input("Dataset Path in HDFS", value="hdfs:///user/oussama/anxiety_papers.json")
        st.caption("Using HDFS protocol explicitly (hdfs:///) helps avoid path resolution issues")

    submit_button = st.button("Submit Question", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # If the question is submitted, call the backend and store the job ID
    if submit_button:
        with st.spinner("Submitting your question..."):
            try:
                response = requests.post(
                    f"{API_ENDPOINT}/submit_question/",
                    json={"text": question, "dataset_path": dataset_path}
                )
                if response.status_code == 200:
                    job_data = response.json()
                    st.session_state.job_id = job_data["job_id"]
                    st.success(f"Question submitted successfully! (Job ID: {st.session_state.job_id[:8]}...)\n\nGenerating answer ...")
                else:
                    st.error(f"Error: {response.text}")
            except Exception as e:
                st.error(f"Connection error: {str(e)}")

    # If a job has been submitted, poll for its result here and display the answer when ready.
    if st.session_state.job_id:
        with st.spinner("Generating answer..."):
            # Polling loop: update every 2 seconds until a final status is returned.
            while True:
                try:
                    status_response = requests.get(f"{API_ENDPOINT}/job_status/{st.session_state.job_id}")
                    if status_response.status_code == 200:
                        job_result = status_response.json()
                        status = job_result.get("status", "").lower()
                        # If the answer is generated or job failed, break the polling loop.
                        if status in ["completed", "simulated", "failed"]:
                            break
                    else:
                        st.error("Error retrieving job status.")
                        break
                except Exception as e:
                    st.error(f"Error connecting to backend: {str(e)}")
                    break
                time.sleep(2)
        # Once the result is ready, clear the spinner and display the output.
        if status in ["completed", "simulated"]:
            st.markdown('<div class="card-container success-container">', unsafe_allow_html=True)
            st.markdown('<p class="result-heading">Question:</p>', unsafe_allow_html=True)
            st.markdown(job_result.get("question", "No question available"))
            st.markdown('<p class="result-heading">Answer:</p>', unsafe_allow_html=True)
            st.markdown(job_result.get("answer", "No answer available"))
            processing_time = job_result.get("processing_time")
            if isinstance(processing_time, (int, float)):
                processing_time_str = f"{processing_time:.2f} seconds"
            else:
                processing_time_str = "N/A"
            st.markdown(f"<p class='text-muted'>Processing Time: {processing_time_str} | Completed on: {job_result.get('date', 'Unknown')}</p>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        elif status == "failed":
            st.markdown('<div class="card-container error-container">', unsafe_allow_html=True)
            st.markdown("### ❌ Job Failed")
            st.markdown(f"Error Message: {job_result.get('error', 'No error message provided')}")
            st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<h2 class="section-header">About This Project</h2>',
                unsafe_allow_html=True)
    st.markdown("""
    **Anxiety QA Platform**
    Uses a MapReduce approach alongside small-scale LLMs to answer questions about
    anxiety disorders, treatments, and clinical guidelines.
    """)
    st.markdown("### Key Features")
    st.markdown("""
    - **Guidelines:** FAISS‑backed retrieval over clinical guideline chunks.
    - **Papers:** MapReduce processing of anxiety research abstracts.
    - **Web:** DuckDuckGo snippet search for supplemental info.
    - **Routing:** Dynamic selection of sources per question.
    - **Synthesis:** Coherent answer via LLM chain.
    """)
    st.markdown("### Architecture")
    arch = """
    graph TD
      A[Streamlit UI] --> B[FastAPI Backend]
      B --> C[MapReduce (PySpark/Hadoop)]
      C --> D[HDFS Anxiety Papers]
      B --> E[MongoDB Guidelines]
      B --> F[DuckDuckGo Search]
      C --> G[LangChain & LLM]
      G --> H[Synthesized Answer]
      H --> B --> A
    """
    st.graphviz_chart(arch)
    st.markdown("### Technologies Used")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Backend:**\n- FastAPI\n- PySpark\n- Hadoop")
    with c2:
        st.markdown("**AI:**\n- LangChain\n- Ollama LLM\n- FAISS")
    with c3:
        st.markdown("**Frontend:**\n- Streamlit\n- REST API\n- Interactive UI")