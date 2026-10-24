"""
Policy Rules Evaluator.

Loads policy rule definitions from `config/policy_rules.yaml`
and deterministically evaluates borrower features and ML predictions.
"""

import os
import yaml

class PolicyRuleEvaluator:
    def __init__(self, config_path=None):
        if config_path is None:
            config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "policy_rules.yaml")
            
        with open(config_path, "r") as f:
            data = yaml.safe_load(f)
        self.rules = data.get("rules", [])
        
    def evaluate_rules(self, context_dict):
        """
        Evaluates context dictionary against all configured policy rules.
        Returns list of triggered rules.
        """
        triggered = []
        for r in self.rules:
            cond = r["condition"]
            try:
                # Safe eval with context dictionary
                is_triggered = eval(cond, {"__builtins__": None}, context_dict)
                if is_triggered:
                    triggered.append(r)
            except Exception as e:
                print(f"[WARN] Error evaluating rule '{r['name']}': {e}")
                
        return triggered

def get_policy_evaluator():
    return PolicyRuleEvaluator()
