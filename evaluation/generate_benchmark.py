
import json
from pathlib import Path

POLICIES = [
    ("password_policy", "Password Policy",
     "Employees must change compromised passwords immediately.",
     "When must compromised passwords be changed?"),

    ("backup_policy", "Backup Policy",
     "Critical database backups must be performed every 6 hours.",
     "How frequently are critical database backups performed?"),

    ("data_retention", "Data Retention Policy",
     "Customer transaction records must be retained for 7 years.",
     "How long must customer transaction records be retained?"),

    ("remote_work", "Remote Work Policy",
     "Remote employees must connect through the corporate VPN.",
     "What connection must remote employees use?"),

    ("encryption_policy", "Encryption Policy",
     "Sensitive stored data must use AES-256 encryption.",
     "What encryption standard is required for sensitive stored data?"),

    ("vendor_assessment", "Vendor Assessment Policy",
     "High-risk vendors must undergo security assessments every 6 months.",
     "How often are high-risk vendors assessed?"),

    ("incident_escalation", "Incident Escalation Policy",
     "Critical incidents must be escalated to the security director within 15 minutes.",
     "Who receives critical incident escalations and how quickly?"),

    ("employee_training", "Employee Training Policy",
     "Employees must complete cybersecurity training every 12 months.",
     "How frequently must employees complete cybersecurity training?"),

    ("device_management", "Device Management Policy",
     "Lost corporate devices must be reported within 2 hours.",
     "What is the reporting deadline for lost corporate devices?"),

    ("cloud_security", "Cloud Security Policy",
     "Cloud administrator accounts must use hardware security keys.",
     "What authentication device is required for cloud administrators?"),

    ("audit_logging", "Audit Logging Policy",
     "Security audit logs must be retained for 365 days.",
     "How long must security audit logs be retained?"),

    ("data_classification", "Data Classification Policy",
     "Restricted information must be labeled Confidential.",
     "What label must restricted information receive?"),

    ("api_security", "API Security Policy",
     "External APIs must enforce a limit of 100 requests per minute.",
     "What rate limit applies to external APIs?"),

    ("change_management", "Change Management Policy",
     "Production changes require approval from two reviewers.",
     "How many reviewers must approve production changes?"),

    ("business_continuity", "Business Continuity Policy",
     "Disaster recovery exercises must be conducted twice per year.",
     "How often must disaster recovery exercises be conducted?"),

    ("physical_security", "Physical Security Policy",
     "Visitors must wear identification badges inside restricted facilities.",
     "What must visitors wear in restricted facilities?"),

    ("network_security", "Network Security Policy",
     "Firewall rules must be reviewed every 60 days.",
     "How often must firewall rules be reviewed?"),

    ("hr_onboarding", "HR Onboarding Policy",
     "New employees must complete compliance orientation within 5 business days.",
     "What is the compliance orientation deadline for new employees?"),

    ("financial_controls", "Financial Controls Policy",
     "Expenses exceeding $5000 require finance director approval.",
     "Who must approve expenses exceeding $5000?"),

    ("procurement_policy", "Procurement Policy",
     "Purchases exceeding $10000 require three competitive quotations.",
     "How many quotations are required for purchases exceeding $10000?"),
]


def main():
    data_dir = Path("data/benchmark_v2")
    data_dir.mkdir(parents=True, exist_ok=True)

    questions = []

    for name, title, statement, question in POLICIES:
        filename = f"{name}.txt"

        (data_dir / filename).write_text(
            f"{title}\n\n{statement}\n",
            encoding="utf-8",
        )

        questions.append({
            "question": question,
            "relevant_sources": [filename],
        })

    output = Path("evaluation/questions_v2.json")
    output.write_text(
        json.dumps(questions, indent=2),
        encoding="utf-8",
    )

    print(f"Generated {len(POLICIES)} documents")
    print(f"Generated {len(questions)} questions")


if __name__ == "__main__":
    main()
