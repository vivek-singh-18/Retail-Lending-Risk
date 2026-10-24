"""
Citation Validator and Hallucination Prevention Module.

Enforces strict compliance checks:
1. Validates that every policy claim contains a valid citation tag [Document: ..., Section: ..., Page: ...].
2. Verifies that cited policy documents exist in retrieved policy context.
3. Rejects ungrounded statements and returns an INSUFFICIENT_EVIDENCE status if policy evidence is missing.
"""

import re

CITATION_REGEX = r"\[Document:\s*([^,]+),\s*Section:\s*([^,]+),\s*Page:\s*(\d+)\]"

def validate_citations(generated_text, retrieved_context_list=None):
    """
    Validates citations in generated text against retrieved policy documents.
    """
    citations_found = re.findall(CITATION_REGEX, generated_text)
    
    if not citations_found:
        # Check if the response explicitly states insufficient policy evidence
        if "insufficient policy evidence" in generated_text.lower():
            return {
                "status": "VALID",
                "evidence_status": "INSUFFICIENT_EVIDENCE",
                "citations_count": 0,
                "citations": []
            }
        else:
            return {
                "status": "INVALID",
                "evidence_status": "UNGROUNDED_CLAIM",
                "citations_count": 0,
                "error": "Generated text contains policy claims without proper [Document: ..., Section: ..., Page: ...] citations."
            }
            
    retrieved_docs = set()
    if retrieved_context_list:
        for chunk in retrieved_context_list:
            if isinstance(chunk, dict) and "metadata" in chunk:
                retrieved_docs.add(chunk["metadata"].get("document", ""))
                
    valid_citations = []
    unsupported_citations = []
    
    for doc, sec, page in citations_found:
        doc_name = doc.strip()
        citation_str = f"[Document: {doc_name}, Section: {sec.strip()}, Page: {page.strip()}]"
        
        if retrieved_context_list and doc_name not in retrieved_docs:
            unsupported_citations.append(citation_str)
        else:
            valid_citations.append(citation_str)
            
    if unsupported_citations:
        return {
            "status": "INVALID",
            "evidence_status": "HALLUCINATED_CITATION",
            "citations_count": len(valid_citations),
            "unsupported_citations": unsupported_citations,
            "error": f"Citations reference documents not present in retrieved context: {unsupported_citations}"
        }
        
    return {
        "status": "VALID",
        "evidence_status": "SUPPORTED",
        "citations_count": len(valid_citations),
        "citations": valid_citations
    }
