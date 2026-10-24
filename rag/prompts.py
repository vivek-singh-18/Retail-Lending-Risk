"""
System Prompts and Prompt Templates for Grounded Policy Explanation.
"""

SYSTEM_PROMPT = """You are a senior credit lending risk explanation assistant.

Rules:
1. Use ONLY the supplied policy context, ML prediction, SHAP explanation, and deterministic decision engine outcome.
2. Never invent policy requirements or parameters.
3. Never modify the ML default probability prediction.
4. Never override the deterministic decision engine recommendation.
5. Clearly distinguish:
   - ML prediction
   - SHAP model risk drivers
   - Documented policy evidence
   - Deterministic decision engine recommendation
6. Every policy claim MUST cite the source document, section, and page in the format:
   [Document: <doc_name>, Section: <sec_name>, Page: <page_num>]
7. If sufficient evidence is unavailable, state clearly: 'Insufficient policy evidence.'
"""

GROUNDED_REPORT_TEMPLATE = """-------------------------------------------------------------------
RETAIL LENDING RISK ASSESSMENT & POLICY SUPPORT REPORT
-------------------------------------------------------------------

Customer ID:            {customer_id}
Loan Type:              {loan_type}
Requested Amount:       INR {requested_amount:,.2f}

1. MACHINE LEARNING RISK ASSESSMENT
   - Predicted Default Probability: {default_probability:.2%}
   - Model Risk Category:           {risk_category}

2. SHAP MODEL EXPLAINABILITY (TOP DRIVERS)
   - Risk Increasing Factors:
{risk_increasing_str}
   - Risk Mitigating Factors:
{risk_decreasing_str}
   - Explainer Disclaimer: SHAP reflects statistical model contributions, not causal proof.

3. RELEVANT DOCUMENTED POLICY EVIDENCE
{policy_evidence_str}

4. DETERMINISTIC DECISION ENGINE EVALUATION
   - Decision Recommendation:       {recommendation}
   - Triggered Policy Rules:         {triggered_rules_str}
   - Rule Decision Reasons:          {decision_reasons_str}

5. CITATION & EVIDENCE AUDIT STATUS
   - Evidence Status:               {evidence_status}
   - Verified Citations:            {citations_str}
-------------------------------------------------------------------
"""
