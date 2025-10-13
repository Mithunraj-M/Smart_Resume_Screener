# node_process_resume.py
import os
import pprint
import re
import json

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import pinecone
from PyPDF2 import PdfReader
import google.generativeai as genai 

from .state import GraphState
from .process_jd import process_job_description

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") 
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

genai.configure(api_key=GOOGLE_API_KEY)

print("Initializing models and connections for Node 2...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL)
generative_model = genai.GenerativeModel('gemini-2.5-flash')

try:
    pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)
    print("Pinecone connection successful for Node 2.")
except Exception as e:
    print(f"Error initializing Pinecone in Node 2: {e}")
    index = None

WEIGHTS = {
    "semantic_match": 0.5,
    "skill_match": 0.5
}


def extract_text_from_pdf(pdf_path: str = None, pdf_content: bytes = None) -> str:
    """Extracts text from PDF file path or PDF content bytes."""
    text = ""
    try:
        if pdf_content:
            # Handle bytes input (from uploaded file)
            import io
            reader = PdfReader(io.BytesIO(pdf_content))
        else:
            # Handle file path input (existing functionality)
            reader = PdfReader(pdf_path)
            
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    except Exception as e:
        print(f"Error reading PDF: {e}")
    return text

def extract_skills_with_llm(text: str, text_type: str = "resume") -> list:
    """
    Uses LLM to extract ALL relevant technical skills from text.
    No constraints - LLM can identify any skill it finds.
    """
    print(f"\n>>> Extracting skills from {text_type} using LLM...")

    prompt = f"""
    You are an expert technical recruiter and HR analyst. Your task is to extract ALL technical skills, technologies, frameworks, and tools mentioned in the following {text_type} text.

    {text_type.title()} Text:
    ---
    {text[:3000]}  # Limit to first 3000 chars to avoid token limits
    ---

    Please extract ALL relevant technical skills including:
    - Programming languages (e.g., Python, Java, JavaScript, C++, Go, Rust)
    - Frameworks and libraries (e.g., React, Django, Spring Boot, TensorFlow)
    - Databases and storage (e.g., PostgreSQL, MongoDB, Redis, Elasticsearch)
    - Cloud platforms and services (e.g., AWS, Azure, GCP, Docker, Kubernetes)
    - DevOps and deployment tools (e.g., Jenkins, Terraform, Git, CI/CD)
    - Methodologies and practices (e.g., Agile, Scrum, TDD, Microservices)
    - Domain-specific technologies (e.g., Blockchain, AI/ML, IoT, Embedded Systems)

    Return ONLY a valid JSON array of strings, with no other text before or after it.
    Be comprehensive but avoid duplicates. Use lowercase for consistency.

    Example: ["python", "react", "aws", "docker", "kubernetes", "postgresql", "agile", "microservices"]
    """

    try:
        response = generative_model.generate_content(prompt)
        cleaned_response = response.text.strip().replace("```json", "").replace("```", "")
        skills = json.loads(cleaned_response)
        
        # Ensure we have a list and remove duplicates
        if isinstance(skills, list):
            skills = list(set(skills))  # Remove duplicates
            print(f"<<< LLM identified {len(skills)} skills from {text_type}.")
            return skills
        else:
            print(f"!!! LLM returned non-list format: {skills}")
            return []
            
    except Exception as e:
        print(f"!!! LLM skill extraction failed for {text_type}: {e}")
        return []



def process_and_score_resume(state: GraphState) -> GraphState:
    """
    Processes a resume, extracts features, and scores it against the JD.
    Now accepts resume content directly from state.
    """
    print("\n--- Executing Node: process_and_score_resume ---")
    if index is None:
        raise ConnectionError("Pinecone index not initialized for Node 2.")

    # Get resume content from state instead of file path
    resume_content = state.get("resume_content")  # bytes
    resume_path = state.get("resume_path")        # string path (for testing)
    
    if resume_content:
        resume_text = extract_text_from_pdf(pdf_content=resume_content)
        print("Successfully extracted text from resume content")
    elif resume_path:
        resume_text = extract_text_from_pdf(pdf_path=resume_path)
        print(f"Successfully extracted text from resume: {resume_path}")
    else:
        raise ValueError("Either resume_content or resume_path must be provided")
    
    if not resume_text:
        raise ValueError("Could not extract text from resume")

    # Store resume text in state for later use
    state["resume_text"] = resume_text

    # Extract skills from resume using LLM
    resume_skills = extract_skills_with_llm(resume_text, "resume")
    extracted_features = {"skills": resume_skills}
    print(f"Found {len(resume_skills)} skills in resume: {resume_skills}")

    # Calculate semantic match score
    resume_embedding = embedding_model.encode(resume_text).tolist()
    query_response = index.query(vector=resume_embedding, top_k=3, include_metadata=False)
    semantic_scores = [match['score'] for match in query_response['matches']]
    semantic_match_score = sum(semantic_scores) / len(semantic_scores) if semantic_scores else 0
    print(f"Semantic match score (avg of top 3): {semantic_match_score:.4f}")

    # Extract required skills from job description using LLM
    jd_text = state["job_description"]
    required_skills = extract_skills_with_llm(jd_text, "job description")
    
    # Calculate skill match score
    matched_skills = set()
    if not required_skills:
        skill_match_score = 0
        print("Warning: No skills identified in job description. Skill score is 0.")
    elif not resume_skills:
        skill_match_score = 0
        print("Warning: No skills identified in resume. Skill score is 0.")
    else:
        # Find matches between resume and required skills (case-insensitive)
        resume_skills_lower = [skill.lower().strip() for skill in resume_skills]
        required_skills_lower = [skill.lower().strip() for skill in required_skills]
        
        matched_skills = set(required_skills_lower).intersection(set(resume_skills_lower))
        
        # Convert back to original case for display
        matched_skills_original = []
        for req_skill in required_skills:
            if req_skill.lower().strip() in matched_skills:
                matched_skills_original.append(req_skill)
        
        skill_match_score = len(matched_skills) / len(required_skills)
        print(f"Required skills (from LLM): {required_skills}")
        print(f"Resume skills (from LLM): {resume_skills}")
        print(f"Matched skills: {matched_skills_original}")
        print(f"Skill match score: {len(matched_skills)}/{len(required_skills)} = {skill_match_score:.4f}")

    overall_score = (
        semantic_match_score * WEIGHTS["semantic_match"] +
        skill_match_score * WEIGHTS["skill_match"]
    )
    print(f"Overall weighted score: {overall_score:.4f}")

    state["extracted_resume_features"] = extracted_features
    state["scores"] = {
        "overall_score": round(overall_score, 4),
        "semantic_match_score": round(semantic_match_score, 4),
        "skill_match_score": round(skill_match_score, 4),
        "matched_skills": sorted(list(matched_skills)),
        "required_skills": required_skills
    }
    
    return state


if __name__ == "__main__":
    print("\n--- Starting Full Workflow Test (Node 1 -> Node 2 with LLM Query Translation) ---")
    
    JD_PDF_PATH = os.path.join("data", "raw", "Job Description by APTIV - 2025-2026.pdf")
    RESUME_PDF_PATH = os.path.join("data", "raw", "Mithunraj_SDE_updated.pdf")

    if not os.path.exists(JD_PDF_PATH) or not os.path.exists(RESUME_PDF_PATH):
        print(f"\nERROR: Ensure PDF files exist at {JD_PDF_PATH} and {RESUME_PDF_PATH}")
        exit()
        
    jd_text = extract_text_from_pdf(JD_PDF_PATH)
    
    initial_state: GraphState = {
        "job_description": jd_text,
        "resume_path": RESUME_PDF_PATH,
        "resume_content": None,
        "resume_text": "",
        "candidate_name": "",
        "jd_chunks": [],
        "extracted_resume_features": {},
        "scores": {},
        "final_summary": ""
    }
    
    state_after_jd = process_job_description(initial_state)
    final_state = process_and_score_resume(state_after_jd)
    
    print("\n\n--- FINAL WORKFLOW OUTPUT ---")
    print("\n** Final State after both nodes: **")
    pprint.pprint(final_state)