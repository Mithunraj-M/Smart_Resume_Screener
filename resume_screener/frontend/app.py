import streamlit as st
import requests
import json
from typing import Dict, Any
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
        color: #1f2937;
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
        2. Upload a candidate's resume PDF
        3. Click 'Analyze Resume' to get results
        4. View the analysis and recommendations
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
        st.subheader("Candidate Resume")
        resume_file = st.file_uploader(
            "Upload Resume PDF",
            type=['pdf'],
            key="resume_upload",
            help="Upload the candidate's resume as a PDF file"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Analyze button
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col2:
        analyze_button = st.button(
            "Analyze Resume",
            type="primary",
            use_container_width=True,
            disabled=(jd_file is None or resume_file is None)
        )
    
    # Results section
    if analyze_button and jd_file and resume_file:
        with st.spinner("Analyzing resume... This may take a few moments."):
            # Reset file pointers
            jd_file.seek(0)
            resume_file.seek(0)
            
            # Call API
            result = call_api(jd_file, resume_file)
        
        if result:
            st.markdown('<div class="result-section">', unsafe_allow_html=True)
            
            # Get candidate name from API response
            candidate_name = result.get('candidate_name', 'Unknown Candidate')
            
            # Display candidate name
            st.markdown(f'<div class="candidate-name">Candidate: {candidate_name}</div>', unsafe_allow_html=True)
            
            # Display scores
            scores = result.get('scores', {})
            overall_score = scores.get('overall_score', 0)
            semantic_score = scores.get('semantic_match_score', 0)
            skill_score = scores.get('skill_match_score', 0)
            
            # Score metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                score_class = get_score_color(overall_score)
                score_label = get_score_label(overall_score)
                st.metric(
                    "Overall Score",
                    f"{overall_score:.2f}",
                    help=f"Overall compatibility: {score_label}"
                )
                st.markdown(f'<p class="{score_class}">{score_label}</p>', unsafe_allow_html=True)
            
            with col2:
                st.metric(
                    "Semantic Match",
                    f"{semantic_score:.2f}",
                    help="Conceptual alignment with job requirements"
                )
            
            with col3:
                st.metric(
                    "Skill Match",
                    f"{skill_score:.2f}",
                    help="Direct match of required skills"
                )
            
            st.markdown("---")
            
            # Skills analysis
            matched_skills = scores.get('matched_skills', [])
            required_skills = scores.get('required_skills', [])
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Required Skills")
                if required_skills:
                    for skill in required_skills:
                        if skill in matched_skills:
                            st.markdown(f"✅ {skill}")
                        else:
                            st.markdown(f"❌ {skill}")
                else:
                    st.info("No specific skills identified")
            
            with col2:
                st.subheader("Matched Skills")
                if matched_skills:
                    for skill in matched_skills:
                        st.markdown(f"✅ {skill}")
                else:
                    st.info("No skills matched")
            
            st.markdown("---")
            
            # AI Summary
            st.subheader("Analysis Summary")
            summary = result.get('final_summary', 'No summary available')
            st.info(summary)
            
            # File information
            st.markdown("---")
            st.subheader("File Information")
            col1, col2 = st.columns(2)
            with col1:
                st.text(f"Job Description: {result.get('job_filename', 'Unknown')}")
            with col2:
                st.text(f"Resume: {result.get('resume_filename', 'Unknown')}")
            
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
