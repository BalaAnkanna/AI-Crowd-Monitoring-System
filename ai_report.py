# ai_report.py

def generate_report(max_people, total_violation_events):

    if max_people == 0:
        return """
=============================
      SESSION REPORT
=============================
No people detected.
System Status: SAFE
=============================
"""

    if total_violation_events == 0:
        risk = "LOW"
        advice = "No social distancing violations detected."
    elif total_violation_events < 5:
        risk = "MEDIUM"
        advice = "Few violation events detected during session."
    else:
        risk = "HIGH"
        advice = "Multiple violation events detected. Immediate action recommended."

    return f"""
=============================
      SESSION REPORT
=============================
Maximum People in Frame : {max_people}
Unique Violation Events : {total_violation_events}
Risk Level : {risk}

Recommendation:
{advice}
=============================
"""