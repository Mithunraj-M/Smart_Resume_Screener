# node_process_jd.py
import os
import pprint
import pdfplumber
import json
from typing import Dict, List, Any

from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import pinecone
import google.generativeai as genai

from .state import GraphState

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

EMBEDDING_MODEL = "all-MiniLM-L6-v2"  
print("Initializing embedding model...")
model = SentenceTransformer(EMBEDDING_MODEL)
print("Model loaded.")

# Initialize Gemini
genai.configure(api_key=GOOGLE_API_KEY)
generative_model = genai.GenerativeModel('gemini-2.5-flash')

try:
    print("Initializing Pinecone client...")
    pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)
    print("Pinecone index stats:", index.describe_index_stats())
    print("Pinecone index loaded successfully.")
except Exception as e:
    print(f"Error initializing Pinecone: {e}")
    index = None


def translate_jd_to_structured_requirements(jd_text: str) -> Dict[str, Any]:
    """
    Use Gemini LLM to extract structured requirements from job description.
    """
    print("\n>>> Translating JD to structured requirements...")
    
    prompt = f"""
    Analyze this job description and extract structured requirements. Return a JSON object with these keys:
    
    Job Description:
    ---
    {jd_text[:3000]}  # Limit to avoid token limits
    ---
    
    Extract and return as JSON:
    {{
        "required_experience": "years and type of experience needed",
        "hard_skills": ["list of technical skills and technologies"],
        "soft_skills": ["list of interpersonal and behavioral skills"],
        "required_tools": ["specific tools, platforms, frameworks"],
        "education_requirements": "degree and field requirements",
        "certifications": ["required certifications"],
        "project_experience": ["types of projects they should have worked on"],
        "industry_experience": "specific industry or domain experience needed"
    }}
    
    Be specific and comprehensive. If a category is not mentioned, use empty string or array.
    Return ONLY valid JSON, no other text.
    """
    
    try:
        response = generative_model.generate_content(prompt)
        if not response or not response.text:
            print("!!! JD translation failed: Empty response from LLM")
            return get_default_requirements()
            
        cleaned_response = response.text.strip().replace("```json", "").replace("```", "")
        print(f"Raw JD LLM response: {cleaned_response[:500]}...")
        
        requirements = json.loads(cleaned_response)
        print(f"<<< Extracted structured requirements: {list(requirements.keys())}")
        return requirements
    except json.JSONDecodeError as e:
        print(f"!!! JD translation failed - JSON decode error: {e}")
        print(f"Raw response that failed: {cleaned_response if 'cleaned_response' in locals() else 'No response'}")
        return get_default_requirements()
    except Exception as e:
        print(f"!!! JD translation failed: {e}")
        return get_default_requirements()

def get_default_requirements():
    """Return default empty requirements structure."""
    return {
        "required_experience": "",
        "hard_skills": [],
        "soft_skills": [],
        "required_tools": [],
        "education_requirements": "",
        "certifications": [],
        "project_experience": [],
        "industry_experience": ""
    }

def perform_multi_query_search(jd_requirements: Dict[str, Any], resume_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Perform multi-query search for each JD category against resume chunks.
    """
    print("\n--- Performing multi-query search ---")
    
    category_scores = {}
    
    # Define category mappings
    category_mappings = {
        "work_experience": ["work_experience"],
        "hard_skills": ["skills"],
        "projects": ["projects"],
        "education": ["education"],
        "certifications": ["certifications"]
    }
    
    for jd_category, jd_requirements_text in jd_requirements.items():
        if not jd_requirements_text or (isinstance(jd_requirements_text, list) and len(jd_requirements_text) == 0):
            category_scores[jd_category] = {"score": 0.0, "matches": []}
            continue
            
        # Create query text for this category
        if isinstance(jd_requirements_text, list):
            query_text = f"Required {jd_category.replace('_', ' ')}: {', '.join(jd_requirements_text)}"
        else:
            query_text = f"Required {jd_category.replace('_', ' ')}: {jd_requirements_text}"
        
        # Create embedding for query
        query_embedding = model.encode(query_text).tolist()
        
        # Search against resume chunks
        matches = []
        for chunk in resume_chunks:
            # Calculate cosine similarity
            import numpy as np
            similarity = np.dot(query_embedding, chunk['embedding']) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(chunk['embedding'])
            )
            
            matches.append({
                'chunk_id': chunk['chunk_id'],
                'category': chunk['category'],
                'similarity': float(similarity),
                'text': chunk['text']
            })
        
        # Get top 5 matches
        top_matches = sorted(matches, key=lambda x: x['similarity'], reverse=True)[:5]
        
        # Calculate average similarity score
        avg_similarity = sum(match['similarity'] for match in top_matches) / len(top_matches) if top_matches else 0.0
        
        category_scores[jd_category] = {
            "score": round(avg_similarity, 4),
            "matches": top_matches
        }
        
        print(f"{jd_category}: {len(top_matches)} matches, avg similarity: {avg_similarity:.4f}")
        if top_matches:
            print(f"  Top match: {top_matches[0]['text'][:100]}... (similarity: {top_matches[0]['similarity']:.4f})")
    
    return category_scores

def calculate_consolidated_score(category_scores: Dict[str, Any]) -> float:
    """
    Calculate weighted consolidated score.
    """
    weights = {
        "work_experience": 0.4,
        "hard_skills": 0.3,
        "projects": 0.2,
        "education": 0.05,
        "certifications": 0.05
    }
    
    weighted_sum = 0.0
    total_weight = 0.0
    
    for category, data in category_scores.items():
        if category in weights:
            score = data["score"]
            weight = weights[category]
            weighted_sum += score * weight
            total_weight += weight
    
    consolidated_score = weighted_sum / total_weight if total_weight > 0 else 0.0
    return round(consolidated_score, 4)

def process_job_description(state: GraphState) -> GraphState:
    """
    Translate JD to structured requirements and perform multi-query search against resume chunks.
    """
    print("\n--- Executing Node: process_job_description ---")
    if index is None:
        raise ConnectionError("Pinecone index not initialized.")

    jd_text = state["job_description"]
    if not jd_text:
        raise ValueError("Job Description text is empty. Check the PDF extraction.")
    
    # Get resume chunks from state
    resume_chunks = state.get("resume_chunks", [])
    if not resume_chunks:
        raise ValueError("No resume chunks found. Resume processing must happen first.")
    
    # Translate JD to structured requirements only if not already present (batch mode may inject)
    jd_requirements = state.get("jd_requirements") or translate_jd_to_structured_requirements(jd_text)
    state["jd_requirements"] = jd_requirements
    
    # Perform multi-query search
    category_scores = perform_multi_query_search(jd_requirements, resume_chunks)
    state["category_scores"] = category_scores
    
    # Calculate consolidated score
    consolidated_score = calculate_consolidated_score(category_scores)
    state["consolidated_score"] = consolidated_score
    
    print(f"Consolidated score: {consolidated_score}")
    
    return state



if __name__ == "__main__":
    print("\n--- Starting Individual Node Test ---")

    JD_PDF_PATH = r"C:\Users\mithu\Desktop\Smart_Resume_Screener\data\raw\Job Description by APTIV - 2025-2026.pdf"          # <--- CHANGE THIS
    RESUME_PDF_PATH = r"C:\Users\mithu\Desktop\Smart_Resume_Screener\data\raw\Mithunraj_SDE_updated.pdf"  # <--- CHANGE THIS

    jd_text_from_pdf = ""
    try:
        with pdfplumber.open(JD_PDF_PATH) as pdf:
            for page in pdf.pages:
                jd_text_from_pdf += page.extract_text() + "\n"
        print(f"Successfully loaded text from {JD_PDF_PATH}")
    except FileNotFoundError:
        print(f"ERROR: The file was not found at {JD_PDF_PATH}")
        exit()
    except Exception as e:
        print(f"Error reading PDF file: {e}")
        exit()

    initial_state: GraphState = {
        "job_description": jd_text_from_pdf,
        "resume_path": RESUME_PDF_PATH,
        "resume_content": None,
        "resume_text": "",
        "candidate_name": "",
        "jd_chunks": [],
        "extracted_resume_features": {},
        "scores": {},
        "final_summary": ""
    }

    final_state = process_job_description(initial_state)

    print("\n\n--- TEST RUN OUTPUT ---")
    print("\n** 1. Chunks Created: **")
    for i, chunk in enumerate(final_state['jd_chunks']):
        print(f"  Chunk {i+1}: \"{chunk}\"")

    print("\n** 2. Final Updated State (from this node): **")
    pprint.pprint(final_state)