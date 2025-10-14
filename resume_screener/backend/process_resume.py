# node_process_resume.py
import os
import pprint
import re
import json
from typing import Dict, List, Any

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import pinecone
from PyPDF2 import PdfReader
import google.generativeai as genai 

from .state import GraphState
from .process_jd import process_job_description
from .generate_summary import extract_candidate_name

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



def extract_resume_sections(resume_text: str) -> Dict[str, List[str]]:
    """
    Uses LLM to extract structured sections from resume text.
    """
    print("\n>>> Extracting structured sections from resume...")
    
    prompt = f"""
    You are an expert at parsing resumes. Extract the following sections from this resume text and return them as a JSON object.
    
    Resume Text:
    ---
    {resume_text[:4000]}  # Limit to avoid token limits
    ---
    
    Extract these sections and return as JSON:
    {{
        "summary": ["professional summary or objective text"],
        "work_experience": ["each job as separate string - include company, role, duration, key achievements"],
        "projects": ["each project as separate string - include project name, description, technologies used"],
        "skills": ["ALL technical skills, tools, technologies as ONE comprehensive string separated by commas"],
        "education": ["degree, institution, year"],
        "certifications": ["certification name, issuing body, year"]
    }}
    
    Rules:
    - Each work experience entry should be a separate string
    - Each project should be a separate string  
    - Skills should be ONE single string containing ALL technologies like: "Python, Java, React, AWS, Docker, etc."
    - Do NOT split skills into individual items
    - If a section is not found, use empty array []
    - Be comprehensive but concise
    - Preserve important details like company names, technologies, achievements
    
    Return ONLY valid JSON, no other text.
    """
    
    try:
        response = generative_model.generate_content(prompt)
        if not response or not response.text:
            print("!!! Resume section extraction failed: Empty response from LLM")
            return get_default_sections()
            
        cleaned_response = response.text.strip().replace("```json", "").replace("```", "")
        print(f"Raw LLM response: {cleaned_response[:500]}...")
        sections = json.loads(cleaned_response)
        print(f"<<< Extracted {sum(len(v) for v in sections.values())} total chunks from resume sections")
        
        # Debug: Print what we got for skills
        if "skills" in sections:
            print(f"Skills section: {sections['skills']}")
            
            # Fix skills if they're individual characters
            if sections["skills"] and len(sections["skills"]) > 10:
                # If we have many individual characters, combine them
                combined_skills = "".join(sections["skills"])
                sections["skills"] = [combined_skills]
                print(f"Combined skills into single string: {combined_skills[:100]}...")
        
        return sections
    except json.JSONDecodeError as e:
        print(f"!!! Resume section extraction failed - JSON decode error: {e}")
        print(f"Raw response that failed: {cleaned_response if 'cleaned_response' in locals() else 'No response'}")
        return get_default_sections()
    except Exception as e:
        print(f"!!! Resume section extraction failed: {e}")
        return get_default_sections()

def get_default_sections():
    """Return default empty sections structure."""
    return {
        "summary": [],
        "work_experience": [],
        "projects": [],
        "skills": [],
        "education": [],
        "certifications": []
    }

def chunk_resume_by_sections(resume_text: str) -> List[Dict[str, Any]]:
    """
    Parse resume into structured chunks with embeddings.
    """
    print("\n--- Chunking resume by sections ---")
    
    # Extract structured sections
    sections = extract_resume_sections(resume_text)
    
    chunks = []
    chunk_id = 0
    
    for category, content_list in sections.items():
        if not content_list:
            continue
            
        for i, content in enumerate(content_list):
            if content.strip():  # Only process non-empty content
                # Create embedding for this chunk
                embedding = embedding_model.encode(content).tolist()
                
                # Use simple numeric within-resume IDs here; we will prefix with resume_id when storing
                chunk = {
                    "chunk_id": f"chunk_{chunk_id}",
                    "category": category,
                    "text": content,
                    "embedding": embedding
                }
                chunks.append(chunk)
                chunk_id += 1
                
                print(f"Created chunk {chunk_id} for {category}: {content[:50]}...")
    
    print(f"Total chunks created: {len(chunks)}")
    return chunks

def process_and_score_resume(state: GraphState) -> GraphState:
    """
    Processes a resume by chunking it into structured sections and storing in database.
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

    # Extract candidate name from resume text
    candidate_name = extract_candidate_name(resume_text)
    state["candidate_name"] = candidate_name
    print(f"Extracted candidate name: {candidate_name}")

    # Chunk resume into structured sections
    resume_chunks = chunk_resume_by_sections(resume_text)
    
    # Store chunks in Pinecone database
    if resume_chunks:
        vectors_to_upsert = []
        resume_id = state.get("resume_id") or "resume"
        for chunk in resume_chunks:
            # Namespace by resume_id so multiple resumes can coexist
            vector_id = f"{resume_id}:{chunk['chunk_id']}:{chunk['category']}"
            metadata = {
                "text": chunk["text"],
                "category": chunk["category"],
                "source": "resume",
                "resume_id": resume_id,
                "chunk_id": chunk["chunk_id"]
            }
            vectors_to_upsert.append({
                "id": vector_id,
                "values": chunk["embedding"],
                "metadata": metadata
            })
        
        index.upsert(vectors=vectors_to_upsert)
        print(f"Successfully stored {len(vectors_to_upsert)} resume chunks in Pinecone")
    
    # Store chunks in state
    state["resume_chunks"] = resume_chunks
    
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