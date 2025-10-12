
from typing import TypedDict, List, Dict, Any

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        job_description: The initial job description text.
        resume_path: The file path to the candidate's resume.
        jd_chunks: A list of text chunks from the job description.
        extracted_resume_features: Structured data from the resume.
        scores: The calculated scores for the resume.
        final_summary: The final LLM-generated summary.
    """
    job_description: str
    resume_path: str
    jd_chunks: List[str]
    extracted_resume_features: Dict[str, Any]
    scores: Dict[str, Any]
    final_summary: str