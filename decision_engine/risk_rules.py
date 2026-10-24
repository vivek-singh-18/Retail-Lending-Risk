"""
Risk Rules Evaluator.

Evaluates model default probability against risk category thresholds.
"""

import os
import yaml

class RiskRuleEvaluator:
    def __init__(self, config_path=None):
        if config_path is None:
            config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "risk_thresholds.yaml")
            
        with open(config_path, "r") as f:
            data = yaml.safe_load(f)
        self.thresholds = data.get("default_probability", {})
        
    def evaluate_risk(self, prob_default):
        low_max = self.thresholds.get("low_risk_max", 0.15)
        mod_max = self.thresholds.get("moderate_risk_max", 0.35)
        high_max = self.thresholds.get("high_risk_max", 0.60)
        
        if prob_default < low_max:
            return "LOW", "APPROVE"
        elif prob_default < mod_max:
            return "MODERATE", "APPROVE"
        elif prob_default < high_max:
            return "HIGH", "MANUAL_REVIEW"
        else:
            return "VERY_HIGH", "DECLINE"

def get_risk_evaluator():
    return RiskRuleEvaluator()
