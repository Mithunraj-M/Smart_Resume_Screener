import os
from typing import Dict, Any

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import uvicorn
from .graph import app as resume_screener_app
from .process_resume import extract_text_from_pdf

app = FastAPI(
    title="Smart Resume Screener",
    description="AI-powered resume screening system using LangGraph and vector embeddings",
    version="1.0.0"
)

@app.get("/")
async def check():
    return {"status": "live", "message": "Resume Screener API is running"}

class AnalysisResponse(BaseModel):
    scores: Dict[str,Any]
    final_summary: str
    candidate_name: str
    resume_filename: str
    job_filename: str

@app.post("/analyze/", response_model=AnalysisResponse)
async def analyze_resume(job_description: UploadFile = File(...), resume: UploadFile = File(...)):
    # Validate file types
    if not job_description.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Job description must be a PDF file")
    if not resume.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Resume must be a PDF file")
    
    try:
        # Read file contents
        jd_content = await job_description.read()
        resume_content = await resume.read()
        
        # Extract text from job description
        jd_text = extract_text_from_pdf(pdf_content=jd_content)
        if not jd_text:
            raise HTTPException(status_code=400, detail="Could not extract text from job description")
        
        # Prepare input for the graph
        initial_input = {
            "job_description": jd_text,
            "resume_content": resume_content,
            "resume_path": "",  # Empty for API usage
            "resume_text": "",  # Will be populated by process_resume
            "candidate_name": "",  # Will be populated by process_resume
            "resume_chunks": [],
            "jd_requirements": {},
            "category_scores": {},
            "consolidated_score": 0.0,
            "final_summary": ""
        }
        
        # Run the workflow
        final_state = None
        for state in resume_screener_app.stream(initial_input):
            final_state = state
        
        # Extract results
        final_node_state = final_state['generate_summary']
        category_scores = final_node_state.get('category_scores', {})
        consolidated_score = final_node_state.get('consolidated_score', 0.0)
        summary = final_node_state['final_summary']
        candidate_name = final_node_state.get('candidate_name', 'Unknown Candidate')
        
        # Convert category scores to the expected format
        scores = {
            "overall_score": consolidated_score,
            "category_scores": category_scores,
            "consolidated_score": consolidated_score
        }
        
        return AnalysisResponse(
            scores=scores,
            final_summary=summary,
            candidate_name=candidate_name,
            resume_filename=resume.filename,
            job_filename=job_description.filename
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)