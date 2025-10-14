import streamlit as st
import requests
import json
from typing import Dict, Any, List
import io

# Page configuration
st.set_page_config(
    page_title="Smart Resume Screener",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f2937;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #6b7280;
        text-align: center;
        margin-bottom: 3rem;
    }
    .upload-section {
        background-color: #f9fafb;
        padding: 2rem;
        border-radius: 10px;
        border: 2px dashed #d1d5db;
        margin-bottom: 2rem;
    }
    .result-section {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 10px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }
    .score-high {
        color: #059669;
        font-weight: 600;
    }
    .score-medium {
        color: #d97706;
        font-weight: 600;
    }
    .score-low {
        color: #dc2626;
        font-weight: 600;
    }
    .candidate-name {
        font-size: 1.5rem;
        font-weight: 600;
        color: white;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

def call_api(jd_file, resume_file) -> Dict[str, Any]:
    """Call the FastAPI backend to analyze the resume."""
    try:
        # API endpoint
        url = "http://localhost:8000/analyze/"
        
        # Prepare files for upload
        files = {
            'job_description': ('job_description.pdf', jd_file, 'application/pdf'),
            'resume': ('resume.pdf', resume_file, 'application/pdf')
        }
        
        # Make API request
        response = requests.post(url, files=files, timeout=60)
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
            
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to the API server. Please make sure the backend is running on http://localhost:8000")
        return None

def call_api_batch(jd_file, resume_files: List[Any]) -> Dict[str, Any]:
    """Call the FastAPI backend to analyze multiple resumes."""
    try:
        url = "http://localhost:8000/analyze_batch/"
        files = [('job_description', ('job_description.pdf', jd_file, 'application/pdf'))]
        for rf in resume_files:
            files.append(('resumes', (getattr(rf, 'name', 'resume.pdf'), rf, 'application/pdf')))
        response = requests.post(url, files=files, timeout=300)  # 5 minutes timeout
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to the API server. Please make sure the backend is running on http://localhost:8000")
        return None
    except requests.exceptions.Timeout:
        st.error("Request timed out. The analysis is taking longer than expected.")
        return None
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return None
    except requests.exceptions.Timeout:
        st.error("Request timed out. The analysis is taking longer than expected.")
        return None
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return None

def get_score_color(score: float) -> str:
    """Get color class based on score."""
    if score >= 0.7:
        return "score-high"
    elif score >= 0.4:
        return "score-medium"
    else:
        return "score-low"

def get_score_label(score: float) -> str:
    """Get label based on score."""
    if score >= 0.7:
        return "Strong Match"
    elif score >= 0.4:
        return "Moderate Match"
    else:
        return "Weak Match"

def main():
    # Header
    st.markdown('<h1 class="main-header">Smart Resume Screener</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-powered resume analysis and job matching</p>', unsafe_allow_html=True)
    
    # Sidebar for API status
    with st.sidebar:
        st.header("System Status")
        
        # Check API connection
        try:
            response = requests.get("http://localhost:8000/", timeout=5)
            if response.status_code == 200:
                st.success("Backend API: Connected")
            else:
                st.error("Backend API: Error")
        except:
            st.error("Backend API: Disconnected")
        
        st.markdown("---")
        st.markdown("### Instructions")
        st.markdown("""
        1. Upload a job description PDF
        2. Upload one or more candidate resume PDFs
        3. Click 'Analyze Resumes' to get results
        4. View the analysis and recommendations
        
        **Note:** Batch analysis may take 2-5 minutes depending on the number of resumes.
        """)
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        st.subheader("Job Description")
        jd_file = st.file_uploader(
            "Upload Job Description PDF",
            type=['pdf'],
            key="jd_upload",
            help="Upload the job description as a PDF file"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        st.subheader("Candidate Resumes")
        resume_files = st.file_uploader(
            "Upload one or more Resume PDFs",
            type=['pdf'],
            key="resume_upload",
            help="Upload one or multiple candidate resumes as PDF files",
            accept_multiple_files=True
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Analyze button
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col2:
        analyze_button = st.button(
            "Analyze Resumes",
            type="primary",
            use_container_width=True,
            disabled=(jd_file is None or not resume_files)
        )
    
    # Results section
    if analyze_button and jd_file and resume_files:
        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text(f"Starting analysis of {len(resume_files)} resumes...")
        progress_bar.progress(0.1)
        
        with st.spinner(f"Analyzing {len(resume_files)} resumes... This may take 2-5 minutes depending on the number of resumes."):
            jd_file.seek(0)
            for rf in resume_files:
                try:
                    rf.seek(0)
                except Exception:
                    pass
            
            status_text.text("Sending request to backend...")
            progress_bar.progress(0.3)
            
            batch_result = call_api_batch(jd_file, resume_files)
            
            if batch_result:
                progress_bar.progress(1.0)
                status_text.text("Analysis complete!")
            else:
                progress_bar.progress(0.0)
                status_text.text("Analysis failed.")

        if batch_result and 'results' in batch_result:
            st.markdown('<div class="result-section">', unsafe_allow_html=True)
            st.subheader("Candidates (sorted by score)")

            # Table summary
            import pandas as pd
            rows = []
            for item in batch_result['results']:
                rows.append({
                    'Resume ID': item.get('resume_id'),
                    'Candidate': item.get('candidate_name', 'Unknown'),
                    'Score': round(item.get('consolidated_score', 0.0)*100, 1),
                    'Summary': item.get('summary', '')
                })
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True)

            st.markdown("---")
            st.subheader("Details")

            # Drilldown per candidate
            for item in batch_result['results']:
                with st.expander(f"{item.get('candidate_name','Unknown')} • {item.get('resume_id')} • Score: {item.get('consolidated_score',0.0):.2f}"):
                    st.markdown(f"**Summary:** {item.get('summary','')}")
                    cat_scores = item.get('individual_scores', {})
                    if cat_scores:
                        cs_rows = []
                        for cat, data in cat_scores.items():
                            cs_rows.append({
                                'Category': cat.replace('_',' ').title(),
                                'Score': round(data.get('score',0.0)*100,1),
                                'Best': round(data.get('best_similarity', data.get('score',0.0))*100,1) if isinstance(data, dict) else '' ,
                                'Avg': round(data.get('avg_similarity', data.get('score',0.0))*100,1) if isinstance(data, dict) else ''
                            })
                        cs_df = pd.DataFrame(cs_rows)
                        st.dataframe(cs_df, use_container_width=True, hide_index=True)
                    else:
                        st.info("No category scores available.")

            st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown(
        '<div style="text-align: center; color: #6b7280; font-size: 0.9rem;">'
        'Smart Resume Screener - AI-Powered Analysis'
        '</div>',
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
