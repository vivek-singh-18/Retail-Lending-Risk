"""
Semantic Chunking and Document Parsing Module.

Parses PDF policy documents page-by-page using PyPDF, detects sections,
and generates structured chunks with document, section, page, and policy type metadata.
"""

import os
from pypdf import PdfReader

def parse_pdf_document(pdf_path):
    """
    Parses a single PDF file and returns text chunks with metadata.
    """
    reader = PdfReader(pdf_path)
    filename = os.path.basename(pdf_path)
    policy_type = os.path.basename(os.path.dirname(pdf_path))
    
    chunks = []
    current_section = "General Policy"
    
    for page_idx, page in enumerate(reader.pages):
        page_num = page_idx + 1
        text = page.extract_text() or ""
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        
        chunk_lines = []
        for line in lines:
            if line.startswith("Section ") or "Policy:" in line:
                if chunk_lines:
                    chunk_text = "\n".join(chunk_lines)
                    chunks.append({
                        "text": chunk_text,
                        "metadata": {
                            "document": filename,
                            "page": page_num,
                            "section": current_section,
                            "policy_type": policy_type,
                            "product": "All" if "policy" not in filename else filename.replace("_policy.pdf", "").replace("_", " ").title()
                        }
                    })
                    chunk_lines = []
                current_section = line
            else:
                chunk_lines.append(line)
                
        if chunk_lines:
            chunk_text = "\n".join(chunk_lines)
            chunks.append({
                "text": chunk_text,
                "metadata": {
                    "document": filename,
                    "page": page_num,
                    "section": current_section,
                    "policy_type": policy_type,
                    "product": "All" if "policy" not in filename else filename.replace("_policy.pdf", "").replace("_", " ").title()
                }
            })
            
    return chunks

def load_all_policy_chunks(knowledge_base_dir=None):
    if knowledge_base_dir is None:
        knowledge_base_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "knowledge_base")
        
    all_chunks = []
    for root, _, files in os.walk(knowledge_base_dir):
        for file in files:
            if file.endswith(".pdf"):
                full_path = os.path.join(root, file)
                pdf_chunks = parse_pdf_document(full_path)
                all_chunks.extend(pdf_chunks)
                
    print(f"[PARSED] Loaded {len(all_chunks)} semantic chunks from policy documents.")
    return all_chunks
