"""
End-to-End Demonstration Workflow Script.

Demonstrates complete execution:
DATA VALIDATION -> FEATURE ENGINEERING -> ML PREDICTION -> SHAP EXPLAINABILITY ->
HYBRID RAG RETRIEVAL -> RERANKING -> DETERMINISTIC DECISION ENGINE -> GROUNDED REPORT -> CITATION VALIDATION
"""

import sys
import os
import json

from ml.data_validation import run_all_validations
from ml.predict import get_default_predictor
from ml.explain import get_model_explainer
from rag.hybrid_retriever import get_hybrid_retriever
from rag.reranker import get_policy_reranker
from decision_engine.engine import get_decision_engine
from rag.generator import get_report_generator

def run_demo(borrower_dict=None):
    if borrower_dict is None:
        borrower_dict = {
            "customer_id": "C77410",
            "age": 34,
            "city_tier": "Tier_1",
            "employment_type": "Salaried_Private",
            "monthly_income": 55000,
            "requested_amount": 800000,
            "loan_type": "Personal_Loan",
            "credit_score": 690,
            "credit_utilization": 0.52,
            "recent_delinquency": 1,
            "delinquency_count": 2,
            "active_loans": 2,
            "closed_loans": 1,
            "employer_type": "Private_Ltd",
            "income_stability": "Medium",
            "employment_length": 4.5
        }
        
    print("=" * 70, flush=True)
    print("RETAIL LENDING RISK INTELLIGENCE - END-TO-END DEMO WORKFLOW", flush=True)
    print("=" * 70, flush=True)
    
    # Step 1: Data Validation
    print("\n[STEP 1] Data Quality & Integrity Validation...", flush=True)
    passed, _, _ = run_all_validations()
    if not passed:
        print("[WARN] Data quality warnings detected, proceeding with caution.", flush=True)
        
    # Step 2 & 3: Feature Engineering & ML Prediction
    print("\n[STEP 2 & 3] Feature Engineering & Default Risk Prediction...", flush=True)
    predictor = get_default_predictor()
    ml_pred = predictor.predict_single(borrower_dict)
    print(f" - Predicted Default Probability: {ml_pred['default_probability']:.2%}", flush=True)
    print(f" - Model Risk Category:           {ml_pred['risk_category']}", flush=True)
    print(f" - Optimal Decision Threshold:    {ml_pred['threshold']}", flush=True)
    
    # Step 4: SHAP Explainability
    print("\n[STEP 4] Computing SHAP Model Explainability...", flush=True)
    explainer = get_model_explainer()
    shap_explain = explainer.explain_local(borrower_dict)
    print(" - Top Risk Increasing Factors:", flush=True)
    for item in shap_explain['risk_increasing_factors'][:3]:
        print(f"   * {item['feature']:20s} : +{item['shap_value']:.4f}", flush=True)
    print(" - Top Risk Decreasing Factors:", flush=True)
    for item in shap_explain['risk_decreasing_factors'][:3]:
        print(f"   * {item['feature']:20s} : {item['shap_value']:.4f}", flush=True)
        
    # Step 5: RAG Hybrid Retrieval & Reranking
    print("\n[STEP 5] RAG Policy Evidence Search (Dense + BM25 RRF + Reranking)...", flush=True)
    query_text = f"Borrower applying for {borrower_dict['loan_type']} with DTI {ml_pred['features'].get('dti', 0.5):.2f} and recent delinquency {borrower_dict['recent_delinquency']}"
    
    hybrid = get_hybrid_retriever()
    reranker = get_policy_reranker()
    
    meta_filter = {"product": "Personal Loan"} if borrower_dict['loan_type'] == "Personal_Loan" else None
    candidate_chunks = hybrid.retrieve_hybrid(query_text, top_k=10, metadata_filter=meta_filter)
    top_policy_chunks = reranker.rerank(query_text, candidate_chunks, top_k=3) if candidate_chunks else []
    
    print(f" - Retrieved {len(top_policy_chunks)} top precision policy evidence chunks.", flush=True)
    for idx, c in enumerate(top_policy_chunks):
        m = c['metadata']
        print(f"   [{idx+1}] {m['document']} | Section: {m['section']} | Score: {c.get('rerank_score', 0):.4f}", flush=True)
        
    # Step 6: Deterministic Decision Engine
    print("\n[STEP 6] Executing Deterministic Policy Rule Engine...", flush=True)
    engine = get_decision_engine()
    decision_res = engine.evaluate_application(ml_pred['features'], ml_pred['default_probability'])
    print(f" - Decision Recommendation: {decision_res['recommendation']}", flush=True)
    print(f" - Triggered Policy Rules:   {', '.join(decision_res['triggered_rules']) if decision_res['triggered_rules'] else 'None'}", flush=True)
    
    # Step 7: Grounded LLM Report Generation & Citation Validation
    print("\n[STEP 7] Generating Grounded Risk Report & Validating Citations...", flush=True)
    generator = get_report_generator()
    grounded_res = generator.generate_report(borrower_dict, ml_pred, shap_explain, decision_res, top_policy_chunks)
    
    print("\n" + grounded_res["report_text"], flush=True)
    
    val_status = grounded_res["validation"]
    print(f"Citation Validation Audit: Status = {val_status['status']} | Evidence = {val_status['evidence_status']} | Verified Citations = {val_status['citations_count']}", flush=True)
    print("=" * 70, flush=True)
    print("[SUCCESS] End-to-End Retail Lending Risk Support Workflow Complete!", flush=True)

if __name__ == "__main__":
    run_demo()
