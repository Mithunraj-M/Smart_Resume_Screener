# node_generate_summary.py
import os
import pprint
from typing import Dict, Any

from dotenv import load_dotenv
import google.generativeai as genai
from .state import GraphState

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)
generative_model = genai.GenerativeModel('gemini-2.5-pro')

def extract_candidate_name(resume_text: str) -> str:
    """
    Uses LLM to extract the candidate's name from resume text.
    """
    print(">>> Extracting candidate name from resume...")
    
    prompt = f"""
    You are an expert at extracting information from resumes. Your task is to identify the candidate's full name from the following resume text.

    Resume Text:
    ---
    {resume_text[:2000]}  # Limit to first 2000 chars to avoid token limits
    ---

    Please extract the candidate's full name. Look for:
    - Name at the top of the resume
    - Contact information sections
    - Header sections
    - Any clear name indicators

    Return ONLY the full name (first name and last name) in the format "FirstName LastName". 
    If you cannot find a clear name, return "Unknown Candidate".

    Example outputs:
    - "John Smith"
    - "Sarah Johnson"
    - "Unknown Candidate"
    """

    try:
        response = generative_model.generate_content(prompt)
        name = response.text.strip()
        print(f"<<< Extracted candidate name: {name}")
        return name
    except Exception as e:
        print(f"!!! Name extraction failed: {e}")
        return "Unknown Candidate"

def generate_category_specific_summary(category_scores: Dict[str, Any], consolidated_score: float, candidate_name: str) -> str:
    """
    Generate a category-specific summary based on individual scores.
    """
    print("\n>>> Generating category-specific summary...")
    
    # Find the category with the highest score
    best_category = None
    best_score = 0.0
    
    for category, data in category_scores.items():
        if isinstance(data, dict) and "score" in data:
            score = data["score"]
            if score > best_score:
                best_score = score
                best_category = category
    
    # Create detailed score breakdown
    score_breakdown = []
    for category, data in category_scores.items():
        if isinstance(data, dict) and "score" in data:
            score = data["score"]
            score_breakdown.append(f"{category.replace('_', ' ').title()}: {score:.1%}")
    
    # Ensure safe values for prompt
    best_category_label = (best_category.replace('_', ' ').title()) if isinstance(best_category, str) else "Overall Fit"

    prompt = f"""
    You are an expert HR Analyst. Based on these detailed category scores for candidate {candidate_name}, write a one-sentence summary explaining why this candidate is a strong or weak match.
    
    Category Scores:
    {', '.join(score_breakdown)}
    
    Consolidated Score: {consolidated_score:.1%}
    Best Category: {best_category_label} ({best_score:.1%})
    
    Write a single, clear sentence that:
    1. States if this is a "Strong match", "Good match", "Moderate match", or "Weak match"
    2. Mentions the category with the highest score as the primary reason
    3. Be specific about what makes them strong/weak
    
    Scoring Guidelines:
    - Strong match: consolidated score > 0.6 or best category > 0.7
    - Good match: consolidated score > 0.4 or best category > 0.5
    - Moderate match: consolidated score > 0.2 or best category > 0.3
    - Weak match: below moderate thresholds
    
    Example: "Strong match, primarily due to extensive and highly relevant work experience in software development."
    """
    
    try:
        response = generative_model.generate_content(prompt)
        if not response or not response.text:
            print("!!! Category-specific summary generation failed: Empty response from LLM")
            return f"Moderate match based on overall score of {consolidated_score:.1%}."
            
        summary = response.text.strip()
        print(f"<<< Generated category-specific summary: {summary}")
        return summary
    except Exception as e:
        print(f"!!! Category-specific summary generation failed: {e}")
        return f"Moderate match based on overall score of {consolidated_score:.1%}."

def generate_natural_language_summary(state: GraphState) -> GraphState:
    """
    Generate a natural language summary based on structured category scores.
    """
    print("\n--- Executing Node: generate_natural_language_summary ---")

    # Get candidate name
    candidate_name = state.get("candidate_name", "Unknown Candidate")
    
    # Get category scores and consolidated score
    category_scores = state.get("category_scores", {})
    consolidated_score = state.get("consolidated_score", 0.0)
    
    if not category_scores:
        print("Warning: No category scores available. Skipping summary generation.")
        state["final_summary"] = "Could not generate a summary because no category scores were calculated."
        return state

    # Generate category-specific summary
    summary = generate_category_specific_summary(category_scores, consolidated_score, candidate_name)
    state["final_summary"] = summary
    
    print(f"Final summary: {summary}")
    
    return state

if __name__ == "__main__":
    print("\n--- Starting Individual Node Test for Summary Generation ---")

    test_state: GraphState = {
        "job_description": "...",
        "resume_path": "...",
        "resume_content": None,
        "resume_text": "John Smith\nSoftware Engineer\njohn.smith@email.com\n\nExperience:\n- 5 years in software development\n- Python, Java, C++\n- Machine Learning",
        "candidate_name": "",
        "jd_chunks": [],
        "extracted_resume_features": {
            'skills': ['c', 'embedded systems', 'python']
        },
        "scores": {
            'overall_score': 0.85,
            'semantic_match_score': 0.8,
            'skill_match_score': 0.9,
            'matched_skills': ['c', 'embedded systems'],
            'required_skills': ['c', 'embedded systems']
        },
        "final_summary": ""
    }

    final_state = generate_natural_language_summary(test_state)

    print("\n\n--- TEST RUN OUTPUT ---")
    print("\n** Generated Summary: **")
    print(final_state["final_summary"])

    print("\n** Final State: **")
    pprint.pprint(final_state)