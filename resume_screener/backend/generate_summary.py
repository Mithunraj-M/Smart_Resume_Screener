# node_generate_summary.py
import os
import pprint

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

def generate_natural_language_summary(state: GraphState) -> GraphState:
    """
    Uses an LLM to generate a natural language summary based on the scores.
    Also extracts the candidate's name from the resume.
    """
    print("\n--- Executing Node: generate_natural_language_summary ---")

    # Extract candidate name from resume text
    resume_text = state.get("resume_text", "")
    if resume_text:
        candidate_name = extract_candidate_name(resume_text)
        state["candidate_name"] = candidate_name
    else:
        state["candidate_name"] = "Unknown Candidate"
        print("Warning: No resume text available for name extraction.")

    scores = state.get("scores", {})
    if not scores:
        print("Warning: Scores are empty. Skipping summary generation.")
        state["final_summary"] = "Could not generate a summary because no scores were calculated."
        return state

    prompt = f"""
    You are an expert HR Analyst. Your task is to provide a concise, professional summary of a candidate's suitability for a role based on a quantitative analysis.

    Here is the data:
    - Overall Score: {scores.get('overall_score', 'N/A')} (out of 1.0)
    - Semantic Match Score: {scores.get('semantic_match_score', 'N/A')} (conceptual alignment with the job description)
    - Skill Match Score: {scores.get('skill_match_score', 'N/A')} (direct match of required skills)
    - Required Skills for the Role: {scores.get('required_skills', [])}
    - Candidate's Skills that Matched: {scores.get('matched_skills', [])}

    Based on this data, please write a 2-3 sentence summary. Start with a clear recommendation (e.g., "Strongly Recommended," "Good Fit," "Potential Fit," or "Not a good fit"). Then, provide a brief justification for your recommendation, highlighting the candidate's strengths or weaknesses based on the scores.
    """

    try:
        response = generative_model.generate_content(prompt)
        summary = response.text.strip()
        print(">>> LLM-generated summary created successfully.")
        state["final_summary"] = summary
    except Exception as e:
        print(f"!!! LLM summary generation failed: {e}")
        state["final_summary"] = "An error occurred while generating the summary."

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