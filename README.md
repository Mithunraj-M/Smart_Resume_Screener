# Smart Resume Screener
## Overview
The Smart Resume Screener is an AI-powered system that uses LangGraph for workflow orchestration, vector embeddings for semantic matching, and LLM-based analysis for intelligent resume screening. The system consists of Backend API that uses FastAPI, LangGraph Workflow for orchestration, Pincone for Vector Database, gemini-2.5-flash LLM for text processing and analysis.

## Methodology
The system uses a 3-node LangGraph workflow that processes resumes through these stages:

![Resume Screener LangGraph flow structure](workflow_graph.png)

The GraphState maintains the entire workflow state.

## Core Idea

![Core Idea of Resume Screening](image.png)

# Process_Resume node
- The llm analyzes each resume and provides chunks in clean section wise manner. Ex:work experience, skills, projects etc.
- These Chunks are encoded by the 'all-MiniLM-L6-v2' embedding model
- These encoded chunks are stored in Pinecone DB
- Prompt for section wise seperation: " You are an expert technical recruiter and HR analyst. Your task is to extract ALL technical skills, technologies, frameworks, and tools mentioned in the following {text_type} text.

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

    Example: ["python", "react", "aws", "docker", "kubernetes", "postgresql", "agile", "microservices"] "

# Process_JD node 
- In order to get meaningful and structured requirements from the JD, the LLM is used for query translation in order to get structured requirements.
- Then semantic matching is done using cosine similarity and scores are calculated providing weights for each sections. Ex: Projects: 0.4, Skills:0.3 etc.
- Prompt for query translation for structured requirements: " Analyze this job description and extract structured requirements. Return a JSON object with these keys:
    
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
    Return ONLY valid JSON, no other text."

# Generate_Summary node
- The section wise calculate scores and consolidated scores of the resumes are passed to the LLM which analyzes if the candidate is eligible and justifies why he should be eligible
- The input and output is handled through frontend developed using streamlit for initial developmental purposes.
- Prompt for final summary generation with justifications: " You are an expert HR Analyst. Based on these detailed category scores for candidate {candidate_name}, write a one-sentence summary explaining why this candidate is a strong or weak match.
    
    Category Scores:
    {', '.join(score_breakdown)}
    
    Consolidated Score: {consolidated_score:.1%}
    Best Category: {best_category_label} ({best_score:.1%})
    Recommended decision: {recommendation_label}
    
    Write a single, clear sentence that:
    1. States the provided decision exactly (no stronger/weaker wording): "{recommendation_label}"
    2. Mentions the category with the highest score as the primary reason
    3. Be specific about what makes them strong/weak
    
    Example: "Strong match, primarily due to extensive and highly relevant work experience in software development." "

## Project Setup:
- git clone https://github.com/Mithunraj-M/Smart_Resume_Screener.git
- create virtual environment for the project
    python -m venv .venv
- activate environment
    .venv\Scripts\activate
- pip install -r requirements.txt

## Setup .env file:
PINECONE_API_KEY="your_pincone_api_key"
PINECONE_INDEX_NAME="resume-screener" 
GOOGLE_API_KEY="your_google_api_key"