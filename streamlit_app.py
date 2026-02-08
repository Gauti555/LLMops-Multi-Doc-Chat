import streamlit as st
import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
API_URL = "http://localhost:8000"
LANGSMITH_PROJECT_URL = "https://smith.langchain.com/projects"
MLFLOW_URL = "http://localhost:5000"

st.set_page_config(
    page_title="LLMops Multi-Doc Chat",
    page_icon="🤖",
    layout="wide",
)

# Custom CSS for a premium feel
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #4a90e2;
        color: white;
    }
    .stChatFloatingInputContainer {
        padding-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;
    }
    .sources-caption {
        font-size: 0.8rem;
        color: #888;
        margin-top: -0.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.title("🤖 LLMops Multi-Doc Chat")
    st.subheader("Interactive RAG powered by OpenAI & LangSmith")

    # Sidebar for configuration and file upload
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        try:
            status_resp = requests.get(f"{API_URL}/", timeout=2)
            api_status = status_resp.status_code == 200
        except:
            api_status = False
        
        if api_status:
            st.success("✅ Backend Online")
        else:
            st.error("❌ Backend Offline")
            st.info("Run: `uvicorn main:app --host 0.0.0.0 --port 8000`")
        
        st.divider()
        
        st.header("📂 Document Management")
        uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])
        
        if uploaded_file is not None:
            if st.button("🚀 Ingest Document"):
                with st.spinner("Ingesting and indexing..."):
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                    try:
                        response = requests.post(f"{API_URL}/upload-pdf", files=files, timeout=60)
                        if response.status_code == 200:
                            st.success("Successfully ingested!")
                        else:
                            try:
                                error_msg = response.json().get('detail', response.text)
                                st.error(f"Error indexing: {error_msg}")
                            except:
                                st.error(f"Error indexing: {response.text}")
                    except Exception as e:
                        st.error(f"Connection Error: {e}")

        st.divider()
        
        st.header("📊 MLOps Dashboards")
        st.markdown(f"[🔗 Open LangSmith]({LANGSMITH_PROJECT_URL})")
        st.markdown(f"[🔗 Open MLflow]({MLFLOW_URL})")

    # Chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                st.markdown(f'<p class="sources-caption">Sources: {", ".join(message["sources"])}</p>', unsafe_allow_html=True)

    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Searching documents..."):
                try:
                    response = requests.post(f"{API_URL}/chat", json={"question": prompt}, timeout=30)
                    if response.status_code == 200:
                        data = response.json()
                        answer = data.get("answer", "No answer received.")
                        sources = data.get("sources", [])
                        
                        st.markdown(answer)
                        if sources:
                            st.markdown(f'<p class="sources-caption">Sources: {", ".join(sources)}</p>', unsafe_allow_html=True)
                        
                        st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
                    else:
                        error_detail = "Unknown error"
                        try:
                            error_detail = response.json().get('detail', response.text)
                        except:
                            error_detail = response.text
                        st.error(f"Backend Error ({response.status_code}): {error_detail}")
                except Exception as e:
                    st.error(f"Connection Error: {e}")

if __name__ == "__main__":
    main()
