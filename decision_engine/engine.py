"""
Deterministic Decision Engine.

Master decision module combining ML Default Probability, feature checks,
and hard policy constraints. Outputs final loan decision: APPROVE, MANUAL_REVIEW, or DECLINE.
Strictly non-LLM deterministic Python code.
"""

from decision_engine.policy_rules import get_policy_evaluator
from decision_engine.risk_rules import get_risk_evaluator

class DecisionEngine:
    def __init__(self):
        self.policy_evaluator = get_policy_evaluator()
        self.risk_evaluator = get_risk_evaluator()
        
    def evaluate_application(self, borrower_features, default_probability):
        """
        Executes deterministic decision logic.
        """
        context = borrower_features.copy()
        context["default_probability"] = default_probability
        
        # Evaluate policy rules
        triggered_rules = self.policy_evaluator.evaluate_rules(context)
        risk_category, risk_base_recommendation = self.risk_evaluator.evaluate_risk(default_probability)
        
        # Check for hard declines
        declines = [r for r in triggered_rules if r["action"] == "DECLINE"]
        reviews = [r for r in triggered_rules if r["action"] == "MANUAL_REVIEW"]
        
        if len(declines) > 0:
            final_recommendation = "DECLINE"
        elif len(reviews) > 0 or risk_base_recommendation == "MANUAL_REVIEW":
            final_recommendation = "MANUAL_REVIEW"
        else:
            final_recommendation = "APPROVE"
            
        reasons = []
        citations = []
        for r in triggered_rules:
            reasons.append(r["reason"])
            if "citation" in r:
                citations.append(r["citation"])
                
        if not reasons:
            reasons.append(f"Application satisfies all policy standards with predicted default probability of {default_probability:.2%}.")
            citations.append("[Document: loan_eligibility.pdf, Section: 2.1, Page: 1]")
            
        return {
            "recommendation": final_recommendation,
            "risk_category": risk_category,
            "default_probability": round(default_probability, 4),
            "triggered_rules": [r["name"] for r in triggered_rules],
            "decision_reasons": reasons,
            "citations": citations
        }

def get_decision_engine():
    return DecisionEngine()
