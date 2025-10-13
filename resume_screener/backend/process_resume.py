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
from .skills import CURATED_SKILLS
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

def extract_skills_from_resume(text: str, skill_list: list) -> list:
    """Extracts skills from text based on a curated list (simple keyword match)."""
    found_skills = set()
    text_lower = text.lower()
    text_lower = re.sub(r'\s+', ' ', text_lower).strip()
    for skill in skill_list:
        if re.search(r'\b' + re.escape(skill) + r'\b', text_lower):
            found_skills.add(skill)
    return sorted(list(found_skills))


def expand_job_requirements_with_llm(jd_text: str, skill_list: list) -> list:
    """
    Uses a generative LLM to intelligently extract required skills from a JD.
    """
    print("\n>>> Performing advanced query translation with LLM...")

    prompt = f"""
    You are an expert technical recruiter and HR analyst. Your task is to analyze the following job description and identify the most essential skills required for the role.

    You MUST only choose skills from the provided list of `VALID_SKILLS`. Do not invent skills.

    Job Description:
    ---
    {jd_text}
    ---

    VALID_SKILLS:
    {', '.join(skill_list)}
    ---

    Your final output must be a single, valid JSON array of strings, with no other text before or after it.
    Example: ["python", "c++", "agile", "embedded systems"]
    """

    try:
        response = generative_model.generate_content(prompt)
        cleaned_response = response.text.strip().replace("```json", "").replace("```", "")
        required_skills = json.loads(cleaned_response)
        print(f"<<< LLM identified {len(required_skills)} required skills.")
        return required_skills
    except Exception as e:
        print(f"!!! LLM or JSON parsing failed: {e}")
        print("!!! Falling back to simple keyword extraction for JD.")
        return extract_skills_from_resume(jd_text, skill_list)



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

    resume_skills = extract_skills_from_resume(resume_text, CURATED_SKILLS)
    extracted_features = {"skills": resume_skills}
    print(f"Found {len(resume_skills)} skills in resume: {resume_skills}")

    resume_embedding = embedding_model.encode(resume_text).tolist()
    query_response = index.query(vector=resume_embedding, top_k=3, include_metadata=False)
    semantic_scores = [match['score'] for match in query_response['matches']]
    semantic_match_score = sum(semantic_scores) / len(semantic_scores) if semantic_scores else 0
    print(f"Semantic match score (avg of top 3): {semantic_match_score:.4f}")

    jd_text = state["job_description"]
    required_skills = expand_job_requirements_with_llm(jd_text, CURATED_SKILLS)
    
    matched_skills = set()
    if not required_skills:
        skill_match_score = 0
        print("Warning: No skills identified by LLM or fallback. Skill score is 0.")
    else:
        matched_skills = set(required_skills).intersection(set(resume_skills))
        skill_match_score = len(matched_skills) / len(required_skills)
        print(f"Required skills (from LLM): {required_skills}")
        print(f"Matched skills: {list(matched_skills)}")
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