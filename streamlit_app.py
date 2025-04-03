import streamlit as st
import requests
import time
from datetime import datetime

# API endpoint - change if needed
API_ENDPOINT = "http://localhost:8000"

# Page configuration with custom theme
st.set_page_config(
    page_title="Biomedical Literature Analysis",
    page_icon="🧬",
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
    st.markdown("### 🧬 Genetic Analysis System")
    st.markdown("Analyze biomedical literature about genetic mutations using MapReduce and LLMs.")
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
    st.markdown('<h1 class="main-header">Genetic Mutations Analysis</h1>', unsafe_allow_html=True)
    st.markdown("""
    Submit your question about genetic mutations and their impact on cellular function.
    Our system processes the provided biomedical abstracts and generates an answer once processing completes.
    """)

    # Question submission form in a styled card
    st.markdown('<div class="card-container">', unsafe_allow_html=True)
    default_question = (
        "Based on the provided biomedical abstracts, what are the most significant "
        "impacts of genetic mutations on cellular function, and what mechanisms "
        "are most commonly implicated?"
    )
    question = st.text_area("Your Question:", value=default_question, height=150, key="question_input")

    # Advanced options (e.g., specifying the HDFS dataset path)
    with st.expander("Advanced Options"):
        dataset_path = st.text_input("Dataset Path in HDFS", value="hdfs:///user/oussama/biomedical_abstracts.json")
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
    st.markdown('<h2 class="section-header">About This Project</h2>', unsafe_allow_html=True)
    st.markdown("""
    ### Exploring the Impact of Genetic Mutations on Cellular Function through Biomedical Literature Analysis

    This project demonstrates how to use a MapReduce approach along with small-scale Language Models (LLMs)
    to analyze biomedical literature efficiently at scale.
    """)
    st.markdown("#### Key Features:")
    st.markdown("""
    - **Distributed Processing:** Uses Hadoop and Spark for scalable text analysis.
    - **LLM Integration:** Leverages optimized language models via LangChain and Ollama.
    - **MapReduce Pattern:** Applies a classic divide-and-conquer strategy to process large datasets.
    - **Interactive Analysis:** Lets researchers ask questions and view answers seamlessly.
    """)
    st.markdown("#### System Architecture")
    try:
        architecture_diagram = """
        graph TD
            A[User Interface: Streamlit] --> B[Backend: FastAPI]
            B --> C[Hadoop/PySpark]
            C --> D[HDFS: Biomedical Dataset]
            C --> E[LangChain & Together API]
            E --> F[Small-scale LLM]
            C --> G[MapReduce Processing]
            G --> H[Final Aggregated Answer]
            H --> B
            B --> A
        """
        st.graphviz_chart(architecture_diagram)
    except Exception:
        st.image("https://miro.medium.com/max/1400/1*CJe3891Y4HBQ_dZ0TJyKJw.png", caption="MapReduce Architecture (Example)")
        st.caption("Note: This is a generic MapReduce diagram for illustration")
    st.markdown("#### Technologies Used")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Backend:**\n- Apache Hadoop\n- PySpark\n- FastAPI")
    with col2:
        st.markdown("**AI Components:**\n- LangChain\n- Small-scale LLMs via Together API\n- Text processing utilities")
    with col3:
        st.markdown("**Frontend:**\n- Streamlit\n- API Integration\n- Interactive UI components")

if __name__ == "__main__":
    pass