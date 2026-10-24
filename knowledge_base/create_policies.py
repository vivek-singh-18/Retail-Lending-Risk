"""
Educational Policy Generator.

Generates realistic simulated lending policy PDFs in `knowledge_base/`
using ReportLab to enable document chunking and vector search retrieval.
"""

import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

POLICIES = [
    {
        "folder": "eligibility",
        "filename": "loan_eligibility.pdf",
        "title": "Educational Policy: Borrower Eligibility Framework",
        "content": [
            "Section 1: General Income & Age Eligibility",
            "1.1 Minimum Age Requirement: Applicants must be at least 21 years old at application time and no older than 65 years at loan maturity.",
            "1.2 Minimum Monthly Net Income: For Salaried applicants, minimum net monthly income must be INR 25,000 for Tier-1 cities and INR 20,000 for Tier-2/3 cities. For Self-Employed / Business applicants, minimum monthly average income must be INR 40,000.",
            "1.3 Employment Continuity: Salaried individuals require at least 1.0 year of total employment history with at least 6 months at their current employer. Self-employed applicants require 2.0 years of active business operations with audited ITR filings.",
            "Section 2: Credit Score Eligibility Thresholds",
            "2.1 Tier 1 Prime Eligible: Credit Score >= 750 qualifies for standard automated underwriting and preferential interest rate tiers.",
            "2.2 Tier 2 Near-Prime Eligible: Credit Score between 650 and 749 is eligible subject to debt-to-income (DTI) verification.",
            "2.3 Subprime Ineligibility: Credit Score below 600 results in mandatory decline, unless supported by 100% liquid collateral security."
        ]
    },
    {
        "folder": "underwriting",
        "filename": "underwriting_guidelines.pdf",
        "title": "Educational Policy: Underwriting & DTI Standards",
        "content": [
            "Section 1: Debt-to-Income (DTI) Caps",
            "1.1 Maximum Permissible DTI: Total monthly debt obligations (including requested EMI) must not exceed 50% of net monthly income for standard personal loans.",
            "1.2 High Income DTI Exception: For applicants earning above INR 150,000 monthly, max DTI may extend to 60% subject to Senior Underwriter approval.",
            "1.3 Hard DTI Cutoff: Any application with calculated DTI exceeding 65% must be DECLINED without exception due to excessive debt leverage.",
            "Section 2: Loan-to-Income (LTI) Limits",
            "2.1 Personal Loans: Maximum unsecured loan amount capped at 10x monthly net income.",
            "2.2 Car Loans: Capped at 8x monthly net income or 85% of vehicle invoice value.",
            "2.3 Housing Loans: Capped at 40x monthly net income with maximum LTV ratio of 80%."
        ]
    },
    {
        "folder": "delinquency",
        "filename": "delinquency_policy.pdf",
        "title": "Educational Policy: Delinquency & Credit Bureau Rules",
        "content": [
            "Section 1: Historical Delinquency Impact",
            "1.1 Recent Delinquency Trigger: Any 30+ DPD (Days Past Due) instance within the last 6 months mandates MANUAL_REVIEW by Risk Operations.",
            "1.2 Critical Delinquency Prohibition: Any applicant with a 60+ or 90+ DPD record in the past 12 months or active write-off status must be DECLINED.",
            "1.3 Excessive Delinquency Frequency: Applicants with more than 3 total delinquency events over their credit history must undergo enhanced manual underwriting."
        ]
    },
    {
        "folder": "documentation",
        "filename": "documentation_requirements.pdf",
        "title": "Educational Policy: Verification & KYC Requirements",
        "content": [
            "Section 1: Mandatory Documentation Checklist",
            "1.1 Identity & Address Proof: Valid PAN card along with Aadhaar, Passport, or Voter ID.",
            "1.2 Salaried Proof: Latest 3 months salary slips, 6 months bank statement showing salary credit, and Form 16.",
            "1.3 Self-Employed Proof: 2 years audited Profit & Loss statement, Balance Sheet, 12 months bank statements, and GST returns.",
            "1.4 Discrepancy Protocol: Any mismatch between declared income and bank statement credits exceeding 15% requires MANUAL_REVIEW."
        ]
    },
    {
        "folder": "risk_policies",
        "filename": "risk_review_policy.pdf",
        "title": "Educational Policy: Risk-Based Pricing & Escalation Policy",
        "content": [
            "Section 1: Risk Categorization & Approval Matrix",
            "1.1 LOW Risk (ML Default Prob < 15%): Automated instant APPROVE provided all hard policy rules are satisfied.",
            "1.2 MODERATE Risk (ML Default Prob 15% - 35%): Standard approval with regular underwriting review.",
            "1.3 HIGH Risk (ML Default Prob 35% - 60%): Requires MANUAL_REVIEW by Credit Risk Manager.",
            "1.4 VERY HIGH Risk (ML Default Prob > 60%): Mandatory DECLINE."
        ]
    },
    {
        "folder": "product_policies",
        "filename": "personal_loan_policy.pdf",
        "title": "Educational Policy: Personal Loan Product Rules",
        "content": [
            "Section 1: Personal Loan Specific Conditions",
            "1.1 Maximum Unsecured Exposure: Maximum loan amount INR 1,500,000 for personal loans.",
            "1.2 Minimum Credit Score: Credit Score must be >= 650 for personal loan approval.",
            "1.3 Multiple Active Loans: Applicants holding >= 3 active unsecured personal loans require manual review for debt consolidation assessment."
        ]
    },
    {
        "folder": "product_policies",
        "filename": "housing_loan_policy.pdf",
        "title": "Educational Policy: Housing Loan Product Rules",
        "content": [
            "Section 1: Housing Loan Specific Conditions",
            "1.1 Property Legal & Technical Valuation: All housing loans require satisfactory legal title search report and property valuation by an empanelled valuer.",
            "1.2 Maximum LTV: Loan-to-Value ratio cannot exceed 80% of property market value.",
            "1.3 Tenure Range: 120 to 240 months."
        ]
    },
    {
        "folder": "product_policies",
        "filename": "used_car_loan_policy.pdf",
        "title": "Educational Policy: Used Car Loan Product Rules",
        "content": [
            "Section 1: Vehicle Loan Specific Conditions",
            "1.1 Vehicle Age Limit: Car age must not exceed 7 years at loan maturity.",
            "1.2 Hypothecation: Vehicle must be hypothecated to the lender with RTO endorsement."
        ]
    }
]

def generate_all_pdfs():
    base_dir = os.path.dirname(__file__)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('TitleStyle', parent=styles['Heading1'], fontSize=16, leading=20)
    body_style = ParagraphStyle('BodyStyle', parent=styles['Normal'], fontSize=10, leading=14)
    
    for item in POLICIES:
        folder_path = os.path.join(base_dir, item["folder"])
        os.makedirs(folder_path, exist_ok=True)
        pdf_path = os.path.join(folder_path, item["filename"])
        
        doc = SimpleDocTemplate(pdf_path, pagesize=letter)
        story = []
        
        story.append(Paragraph(item["title"], title_style))
        story.append(Spacer(1, 15))
        
        for line in item["content"]:
            story.append(Paragraph(line, body_style))
            story.append(Spacer(1, 8))
            
        doc.build(story)
        print(f" - Generated policy PDF: {pdf_path}")
        
    print("[SUCCESS] All educational policy documents created successfully!")

if __name__ == "__main__":
    generate_all_pdfs()
