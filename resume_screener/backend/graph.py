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
workflow.add_node("process_resume", process_and_score_resume)
workflow.add_node("process_jd", process_job_description)
workflow.add_node("generate_summary", generate_natural_language_summary)

workflow.set_entry_point("process_resume")

workflow.add_edge("process_resume", "process_jd")
workflow.add_edge("process_jd", "generate_summary")
workflow.add_edge("generate_summary", END)

print("Compiling the graph...")
app = workflow.compile()
print("Graph compiled successfully.")

# Function to visualize the graph structure
def visualize_graph():
    """Generate and display the graph visualization."""
    try:
        from IPython.display import Image, display
        from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles
        
        # Generate the graph image
        graph_image = app.get_graph().draw_mermaid_png()
        
        # Display in Jupyter notebook
        display(Image(graph_image))
        
        # Save to reports/figures directory
        import os
        figures_dir = os.path.join("reports", "figures")
        os.makedirs(figures_dir, exist_ok=True)
        
        png_path = os.path.join(figures_dir, "workflow_graph.png")
        with open(png_path, "wb") as f:
            f.write(graph_image)
        print(f"Graph visualization saved as '{png_path}'")
        
    except ImportError:
        print("IPython not available. Install with: pip install ipython")
    except Exception as e:
        print(f"Error generating graph visualization: {e}")

# Function to save PNG without IPython dependency
def save_graph_png():
    """Save the graph as PNG to reports/figures directory."""
    try:
        # Generate the graph image
        graph_image = app.get_graph().draw_mermaid_png()
        
        # Save to reports/figures directory
        import os
        figures_dir = os.path.join("reports", "figures")
        os.makedirs(figures_dir, exist_ok=True)
        
        png_path = os.path.join(figures_dir, "workflow_graph.png")
        with open(png_path, "wb") as f:
            f.write(graph_image)
        print(f"✅ Graph visualization saved as '{png_path}'")
        return png_path
        
    except Exception as e:
        print(f"❌ Error generating graph visualization: {e}")
        return None

# Alternative: Print the Mermaid diagram as text
def print_mermaid_diagram():
    """Print the Mermaid diagram as text."""
    try:
        mermaid_code = app.get_graph().draw_mermaid()
        print("\n" + "="*50)
        print("MERMAID DIAGRAM CODE:")
        print("="*50)
        print(mermaid_code)
        print("="*50)
        print("Copy this code to https://mermaid.live/ to visualize")
    except Exception as e:
        print(f"Error generating Mermaid diagram: {e}")

if __name__ == "__main__":
    print("\n--- VISUALIZING GRAPH STRUCTURE ---")
    print_mermaid_diagram()
    
    print("\n--- SAVING GRAPH PNG ---")
    save_graph_png()
    
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