# graph.py
import os
import pprint
from langgraph.graph import StateGraph, END

from .state import GraphState
from .process_jd import process_job_description
from .process_resume import process_and_score_resume, extract_text_from_pdf
from .generate_summary import generate_natural_language_summary

print("Defining the graph structure...")
workflow = StateGraph(GraphState)
workflow.add_node("process_jd", process_job_description)
workflow.add_node("process_resume", process_and_score_resume)
workflow.add_node("generate_summary", generate_natural_language_summary)

workflow.set_entry_point("process_jd")

workflow.add_edge("process_jd", "process_resume")
workflow.add_edge("process_resume", "generate_summary")
workflow.add_edge("generate_summary", END)

print("Compiling the graph...")
app = workflow.compile()
print("Graph compiled successfully.")

if __name__ == "__main__":
    print("\n--- RUNNING THE FULL END-TO-END WORKFLOW ---")

    JD_PDF_PATH = os.path.join("data", "raw", "Job Description by APTIV - 2025-2026.pdf")
    RESUME_PDF_PATH = os.path.join("data", "raw", "Mithunraj_SDE_updated.pdf")

    if not os.path.exists(JD_PDF_PATH) or not os.path.exists(RESUME_PDF_PATH):
        print(f"\nERROR: Ensure PDF files exist at {JD_PDF_PATH} and {RESUME_PDF_PATH}")
        exit()

    jd_text = extract_text_from_pdf(JD_PDF_PATH)
    initial_input = {
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

    print("\nInvoking the graph with initial input...")
    final_state = None
    for s in app.stream(initial_input):
        final_state = s
        print("\n" + "="*40)
        print(f"State after node: '{list(s.keys())[0]}'")
        print("="*40)
    print("\n\n--- FINAL WORKFLOW OUTPUT ---")
    print("\n** Final State: **")
    pprint.pprint(final_state)

    print("\n\n**  Final AI-Generated Summary: **")
    print(final_state['generate_summary']['final_summary'])