# node_process_jd.py
import os
import pprint
import pdfplumber

from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import pinecone

from state import GraphState

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

EMBEDDING_MODEL = "all-MiniLM-L6-v2"  
print("Initializing embedding model...")
model = SentenceTransformer(EMBEDDING_MODEL)
print("Model loaded.")

try:
    print("Initializing Pinecone client...")
    pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)
    print("Pinecone index stats:", index.describe_index_stats())
    print("Pinecone index loaded successfully.")
except Exception as e:
    print(f"Error initializing Pinecone: {e}")
    index = None


def process_job_description(state: GraphState) -> GraphState:
    """
    Chunks, embeds, and upserts the job description into the vector DB.
    """
    print("\n--- Executing Node: process_job_description ---")
    if index is None:
        raise ConnectionError("Pinecone index not initialized.")

    jd_text = state["job_description"]
    if not jd_text:
        raise ValueError("Job Description text is empty. Check the PDF extraction.")
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_text(jd_text)
    print(f"Split JD into {len(chunks)} chunks.")

    embeddings = model.encode(chunks).tolist()
    print("Created embeddings for all chunks.")

    vectors_to_upsert = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        vector_id = f"jd_chunk_{i}"
        metadata = {"text": chunk, "source": "job_description"}
        vectors_to_upsert.append({"id": vector_id, "values": embedding, "metadata": metadata})

    index.upsert(vectors=vectors_to_upsert)
    print(f"Successfully upserted {len(vectors_to_upsert)} vectors to Pinecone.")

    state['jd_chunks'] = chunks
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