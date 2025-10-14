
from typing import TypedDict, List, Dict, Any, Optional

class GraphState(TypedDict):
    """
    Represents the state of our graph with structured resume analysis.

    Attributes:
        job_description: The initial job description text.
        resume_id: A unique identifier for the resume (for namespacing/results).
        resume_path: The file path to the candidate's resume (for testing).
        resume_content: The resume PDF content as bytes (for API).
        resume_text: The extracted text from the resume.
        candidate_name: The extracted candidate name from the resume.
        resume_chunks: Structured resume chunks with categories and embeddings.
        jd_requirements: Structured job requirements extracted from JD.
        category_scores: Individual scores for each resume category.
        consolidated_score: Final weighted score for the candidate.
        final_summary: The final LLM-generated summary.
    """
    job_description: str
    resume_id: str
    resume_path: str
    resume_content: Optional[bytes]
    resume_text: str
    candidate_name: str
    resume_chunks: List[Dict[str, Any]]
    jd_requirements: Dict[str, Any]
    category_scores: Dict[str, Any]
    consolidated_score: float
    final_summary: str