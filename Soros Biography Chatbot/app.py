"""
CodeBuddy: Natural Language to Code Explanation Generator
Streamlit Frontend Application
"""

import streamlit as st
from streamlit_ace import st_ace
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from models.code_explainer import CodeExplainer
from models.rag_pipeline import RAGPipeline
from utils.code_processor import CodeProcessor
from config import STREAMLIT_CONFIG

# Page configuration
st.set_page_config(
    page_title=STREAMLIT_CONFIG["page_title"],
    page_icon=STREAMLIT_CONFIG["page_icon"],
    layout=STREAMLIT_CONFIG["layout"],
    initial_sidebar_state=STREAMLIT_CONFIG["initial_sidebar_state"]
)

# Initialize session state
if 'code_explainer' not in st.session_state:
    st.session_state.code_explainer = None
if 'rag_pipeline' not in st.session_state:
    st.session_state.rag_pipeline = None
if 'code_processor' not in st.session_state:
    st.session_state.code_processor = None

def initialize_models():
    """Initialize the AI models"""
    try:
        with st.spinner("Loading AI models..."):
            if st.session_state.code_explainer is None:
                st.session_state.code_explainer = CodeExplainer()
            if st.session_state.rag_pipeline is None:
                st.session_state.rag_pipeline = RAGPipeline()
            if st.session_state.code_processor is None:
                st.session_state.code_processor = CodeProcessor()
        st.success("Models loaded successfully!")
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.info("Please check the model files and try again.")

def main():
    """Main application function"""
    
    # Header
    st.title("🤖 CodeBuddy")
    st.subheader("Natural Language to Code Explanation Generator")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("Settings")
        
        # Model initialization
        if st.button("Initialize Models"):
            initialize_models()
        
        # Model status
        if st.session_state.code_explainer:
            st.success("✅ Code Explainer: Ready")
        else:
            st.warning("⚠️ Code Explainer: Not Loaded")
            
        if st.session_state.rag_pipeline:
            st.success("✅ RAG Pipeline: Ready")
        else:
            st.warning("⚠️ RAG Pipeline: Not Loaded")
        
        st.markdown("---")
        
        # About section
        st.header("About")
        st.markdown("""
        CodeBuddy uses advanced AI models to:
        - Explain Python code in plain English
        - Answer questions about code
        - Find similar code examples
        
        Built with CodeT5 and RAG technology.
        """)
    
    # Main content area
    tab1, tab2, tab3 = st.tabs(["Code Explanation", "Code Q&A", "Code Search"])
    
    with tab1:
        st.header("📝 Code Explanation")
        st.write("Paste your Python code below to get a natural language explanation.")
        
        # Code input
        code_input = st_ace(
            placeholder="Paste your Python code here...",
            language="python",
            theme="monokai",
            height=300,
            font_size=14,
            show_gutter=True,
            show_print_margin=False,
            wrap=True
        )
        
        if st.button("Generate Explanation", type="primary"):
            if not code_input.strip():
                st.warning("Please enter some code first.")
            elif not st.session_state.code_explainer:
                st.error("Please initialize the models first.")
            else:
                try:
                    with st.spinner("Generating explanation..."):
                        # Process code
                        processed_code = st.session_state.code_processor.process_code(code_input)
                        
                        # Generate explanation
                        explanation = st.session_state.code_explainer.explain_code(processed_code)
                        
                        # Display results
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("Your Code")
                            st.code(code_input, language="python")
                        
                        with col2:
                            st.subheader("Explanation")
                            st.write(explanation)
                            
                            # Download explanation
                            st.download_button(
                                label="Download Explanation",
                                data=explanation,
                                file_name="code_explanation.txt",
                                mime="text/plain"
                            )
                            
                except Exception as e:
                    st.error(f"Error generating explanation: {str(e)}")
    
    with tab2:
        st.header("❓ Code Q&A")
        st.write("Ask questions about your code and get intelligent answers.")
        
        # Code input for Q&A
        qa_code = st_ace(
            placeholder="Paste the code you want to ask questions about...",
            language="python",
            theme="monokai",
            height=200,
            font_size=14,
            show_gutter=True,
            show_print_margin=False,
            wrap=True
        )
        
        # Question input
        question = st.text_input("Ask a question about the code:", placeholder="e.g., What does this function do?")
        
        if st.button("Get Answer", type="primary"):
            if not qa_code.strip() or not question.strip():
                st.warning("Please provide both code and a question.")
            elif not st.session_state.rag_pipeline:
                st.error("Please initialize the models first.")
            else:
                try:
                    with st.spinner("Generating answer..."):
                        # Get answer using RAG pipeline
                        answer = st.session_state.rag_pipeline.get_answer(qa_code, question)
                        
                        # Display results
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("Code Context")
                            st.code(qa_code, language="python")
                            st.subheader("Question")
                            st.write(f"**Q:** {question}")
                        
                        with col2:
                            st.subheader("Answer")
                            st.write(f"**A:** {answer}")
                            
                            # Download Q&A
                            qa_content = f"Code:\n{qa_code}\n\nQuestion: {question}\n\nAnswer: {answer}"
                            st.download_button(
                                label="Download Q&A",
                                data=qa_content,
                                file_name="code_qa.txt",
                                mime="text/plain"
                            )
                            
                except Exception as e:
                    st.error(f"Error generating answer: {str(e)}")
    
    with tab3:
        st.header("🔍 Code Search")
        st.write("Find similar code examples and their explanations.")
        
        # Search query
        search_query = st.text_input("Enter your search query:", placeholder="e.g., sorting algorithms, file handling")
        
        # Code snippet to search for
        search_code = st_ace(
            placeholder="Or paste code to find similar examples...",
            language="python",
            theme="monokai",
            height=150,
            font_size=14,
            show_gutter=True,
            show_print_margin=False,
            wrap=True
        )
        
        if st.button("Search Code", type="primary"):
            if not search_query.strip() and not search_code.strip():
                st.warning("Please provide a search query or code snippet.")
            elif not st.session_state.rag_pipeline:
                st.error("Please initialize the models first.")
            else:
                try:
                    with st.spinner("Searching for similar code..."):
                        # Search for similar code
                        if search_code.strip():
                            results = st.session_state.rag_pipeline.search_similar_code(search_code)
                        else:
                            results = st.session_state.rag_pipeline.search_by_query(search_query)
                        
                        # Display results
                        st.subheader("Search Results")
                        
                        if results:
                            for i, result in enumerate(results[:5]):  # Show top 5 results
                                with st.expander(f"Result {i+1} - Similarity: {result['similarity']:.3f}"):
                                    st.code(result['code'], language="python")
                                    st.write("**Explanation:**", result['explanation'])
                        else:
                            st.info("No similar code found. Try a different query.")
                            
                except Exception as e:
                    st.error(f"Error searching code: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            <p>Built with ❤️ using Streamlit, CodeT5, and RAG technology</p>
            <p>For Natural Engineering Language Course</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
