"""
Grounded Explanation Generator.

Synthesizes ML Risk Prediction, SHAP factors, Retrieved Policy Evidence,
and Deterministic Decision Engine outcome into a grounded risk assessment report.
Validates citations against retrieved policy documents.
"""

import os
from rag.prompts import GROUNDED_REPORT_TEMPLATE
from rag.citation_validator import validate_citations

class GroundedReportGenerator:
    def __init__(self):
        pass
        
    def generate_report(self, borrower_dict, ml_pred, shap_explain, decision_res, policy_chunks):
        """
        Generates structured grounded risk report with validated policy citations.
        """
        c_id = borrower_dict.get("customer_id", "C00000")
        l_type = borrower_dict.get("loan_type", "Personal_Loan")
        req_amt = float(borrower_dict.get("requested_amount", 500000))
        
        # Risk Increasing SHAP
        inc_lines = []
        for item in shap_explain.get("risk_increasing_factors", []):
            inc_lines.append(f"     * {item['feature']}: +{item['shap_value']:.4f}")
        risk_inc_str = "\n".join(inc_lines) if inc_lines else "     * None identified"
        
        # Risk Decreasing SHAP
        dec_lines = []
        for item in shap_explain.get("risk_decreasing_factors", []):
            dec_lines.append(f"     * {item['feature']}: {item['shap_value']:.4f}")
        risk_dec_str = "\n".join(dec_lines) if dec_lines else "     * None identified"
        
        # Policy Evidence
        evidence_lines = []
        citations_collected = []
        
        for idx, chunk in enumerate(policy_chunks[:3]):
            meta = chunk.get("metadata", {})
            doc = meta.get("document", "policy.pdf")
            sec = meta.get("section", "Section 1")
            page = meta.get("page", 1)
            
            cit_tag = f"[Document: {doc}, Section: {sec}, Page: {page}]"
            citations_collected.append(cit_tag)
            
            snippet = chunk.get("text", "").replace("\n", " ")
            if len(snippet) > 180:
                snippet = snippet[:177] + "..."
            evidence_lines.append(f"   [{idx+1}] Document: {doc} (Section: {sec}, Page {page})\n       Evidence: \"{snippet}\"\n       Citation: {cit_tag}")
            
        policy_evidence_str = "\n\n".join(evidence_lines) if evidence_lines else "   Insufficient policy evidence available."
        
        trig_rules = decision_res.get("triggered_rules", [])
        trig_rules_str = ", ".join(trig_rules) if trig_rules else "None (Standard Policy Compliance)"
        
        reasons_lines = [f"     - {r}" for r in decision_res.get("decision_reasons", [])]
        reasons_str = "\n".join(reasons_lines)
        
        report_text = GROUNDED_REPORT_TEMPLATE.format(
            customer_id=c_id,
            loan_type=l_type,
            requested_amount=req_amt,
            default_probability=ml_pred.get("default_probability", 0.0),
            risk_category=ml_pred.get("risk_category", "UNKNOWN"),
            risk_increasing_str=risk_inc_str,
            risk_decreasing_str=risk_dec_str,
            policy_evidence_str=policy_evidence_str,
            recommendation=decision_res.get("recommendation", "MANUAL_REVIEW"),
            triggered_rules_str=trig_rules_str,
            decision_reasons_str=reasons_str,
            evidence_status="SUPPORTED" if policy_chunks else "INSUFFICIENT_EVIDENCE",
            citations_str=", ".join(citations_collected) if citations_collected else "None"
        )
        
        # Citation Validation Check
        val_res = validate_citations(report_text, policy_chunks)
        
        return {
            "report_text": report_text,
            "validation": val_res
        }

def get_report_generator():
    return GroundedReportGenerator()
