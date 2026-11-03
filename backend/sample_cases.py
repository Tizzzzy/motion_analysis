"""Built-in Northern District of Illinois breach-of-contract simulation case
and helpers for custom student-pasted case/complaint text.
"""

from typing import Any, Dict, List, Optional

SAMPLE_NDIL_CONTRACT_CASE: Dict[str, Any] = {
    "id": "ndil_lakeview_cold_chain",
    "title": "Sample Case: Heartland Specialty Foods v. Lakeview Cold Chain Logistics (N.D. Ill.)",
    "short_label": "NDIL Breach of Contract Sample",
    "jurisdiction": "U.S. District Court for the Northern District of Illinois (Illinois substantive contract law)",
    "student_case_text": """=== SUPERVISING PARTNER INTAKE NOTE ===
Matter: Heartland Specialty Foods, Inc. v. Lakeview Cold Chain Logistics, LLC
Court: U.S. District Court for the Northern District of Illinois (Case No. 1:26-cv-04182)
Procedural Posture: Summons and Complaint served 9 days ago; responsive pleading deadline in 12 days.

How the file arrived:
Yesterday evening, Marcus Vance (Director of Warehouse Operations at Lakeview Cold Chain Logistics, LLC) emailed our litigation group forwarding the federal Complaint below and asking us to "file a motion to get this thrown out right away." Lakeview Cold Chain Logistics, LLC is a mid-sized Illinois refrigerated warehousing and regional distribution company headquartered in Bolingbrook, Illinois. Under Lakeview's Operating Agreement on file from prior corporate work, formal engagement, litigation budgets over $15,000, dispositive filings, and any settlement authority rest with CEO Elena Rostova (with CFO sign-off required above $25,000), though Marcus Vance is scheduled to join Elena Rostova for your upcoming client call.

=== EXCERPT OF PLAINTIFF'S COMPLAINT ===
IN THE UNITED STATES DISTRICT COURT
FOR THE NORTHERN DISTRICT OF ILLINOIS, EASTERN DIVISION

HEARTLAND SPECIALTY FOODS, INC., Plaintiff,
v.
LAKEVIEW COLD CHAIN LOGISTICS, LLC, Defendant.
Case No. 1:26-cv-04182

COMPLAINT FOR BREACH OF CONTRACT

1. Plaintiff Heartland Specialty Foods, Inc. ("Heartland"), a Wisconsin corporation with its principal place of business in Madison, Wisconsin, brings this action against Defendant Lakeview Cold Chain Logistics, LLC ("Lakeview"), an Illinois limited liability company, for breach of contract. Diversity jurisdiction exists under 28 U.S.C. § 1332.

2. On March 15, 2024, Heartland and Lakeview entered into a written Master Cold-Storage & Distribution Agreement (the "2024 Master Agreement," which Plaintiff references herein though does not attach as an exhibit) under which Lakeview agreed to store and maintain Heartland's artisan dairy inventory at a continuous temperature between 33°F and 38°F at Lakeview's Bolingbrook Facility (Bay C).

3. Heartland performed all duties, conditions, and obligations required of it under the 2024 Master Agreement, or was excused from doing so.

4. Between August 14, 2026 and August 16, 2026, Lakeview breached the 2024 Master Agreement by failing to maintain proper refrigeration in Bay C, allowing ambient temperatures to rise above 49°F for an extended period and spoiling three commercial pallets of aged specialty cheese products.

5. Lakeview further acted in bad faith and in reckless disregard of industry cold-chain standards by failing to notify Heartland promptly of compressor alarms.

6. As a direct and proximate result of Lakeview's breach, Heartland suffered $68,400 in direct wholesale inventory loss, plus at least $340,000 in lost future profits, damaged retail shelf placement, and reputational harm with regional grocery chains.

COUNT I — BREACH OF CONTRACT (Illinois Law)
7. Heartland incorporates Paragraphs 1 through 6 as though fully set forth herein.
8. The 2024 Master Agreement constitutes a valid and enforceable contract under Illinois law.
9. Heartland complied with its contractual obligations.
10. Lakeview materially breached the 2024 Master Agreement, directly causing damages exceeding $408,400.

WHEREFORE, Plaintiff demands judgment against Defendant in an amount exceeding $408,400, plus pre-judgment interest, attorney's fees, and costs.""",
    "client_simulation_profile": """=== CONFIDENTIAL CLIENT SIMULATION PROFILE (FOR AI ROLEPLAY ONLY — DO NOT DUMP ALL AT ONCE) ===
You are roleplaying Elena Rostova, CEO of Lakeview Cold Chain Logistics, LLC (joined occasionally by Operations Director Marcus Vance if asked about warehouse floor details). Stay naturally in character as a practical, time-pressed business executive. Answer ONLY what the lawyer/student actually asks in each turn! Do not volunteer unasked categories.

1. Decision Authority & Approval Process:
   - Lakeview Cold Chain Logistics, LLC is the corporate client.
   - As CEO, Elena Rostova authorizes legal strategy, motion practice, and settlements up to $25,000. Any settlement or cumulative legal spend above $25,000 requires joint sign-off from CFO David Park and the 3-member Managing Board. Marcus Vance (Operations Director) handles day-to-day facts/records ONLY and cannot authorize filings or settlements.

2. Top Three Ranked Outcomes / Priorities (reveal ranking clearly ONLY if asked to rank/prioritize):
   - #1 (Top Priority): Resolve or contain this dispute cleanly before Lakeview's commercial credit-facility refinancing closes in 75 days (lenders scrutinize open >$250k litigation exposure).
   - #2: Keep total legal fees + payout predictable and within budget (avoid a $60k+ discovery war over a $68k shipment).
   - #3: Protect Lakeview's commercial reputation in the Midwest specialty-grocery network and, ideally, preserve a workable relationship with Heartland (Heartland accounts for ~$190,000/yr in recurring storage revenue, and regional grocers talk to one another).

3. Litigation Budget Constraints:
   - Prepared to spend roughly $15,000–$25,000 total on legal fees if necessary, but spending $20,000 just on a Rule 12(b)(6) motion that merely lets Heartland amend its complaint two months later would be frustrating unless it knocks out the $340,000 consequential-damages claim permanently.

4. Settlement Openness & Conditions:
   - Definitely open to early settlement! Would willingly offer a combination of $30,000–$40,000 cash/insurance contribution plus $25,000 in future cold-storage fee credits, conditioned on mutual confidentiality, no admission of fault, and dismissing the $340,000 consequential damages claim. Not willing to pay $400k+.

5. Timing & Deadlines:
   - Critical business deadline: Bank refinancing underwriting review wraps up in ~75 days (about 2.5 months). A drawn-out 6-month motion-to-dismiss briefing cycle that leaves the $408k claim sitting on the docket during underwriting hurts Lakeview.

6. Risk Tolerance:
   - Strongly prefers cost certainty over rolling the dice in protracted litigation, especially if the $340k lost-profits claim can be neutralized or settled briskly.

7. Publicity & Business Disruption Concerns:
   - Very concerned about public docket mudslinging ("reckless cold-chain failures") spooking two larger dairy distributors currently negotiating 2027 contracts, plus disruption of warehouse supervisors sitting for depositions during peak holiday shipping.

8. What Actually Happened — Supporting Documents AND Unfavorable Documents:
   - Favorable facts/documents:
     (a) Signed 2024 Master Agreement & Section 11.2 Limitation-of-Liability Addendum: Explicitly bars consequential/lost-profit damages and caps recovery at 2x monthly storage fees (~$36,000) unless willful misconduct is proven, and has NO attorney's-fees shifting clause!
     (b) Receiving Dock Log (Aug 12, 2026) & Emails: Two of the three Heartland cheese pallets arrived off Heartland's own carrier truck already warm at 45°F ("hot-loaded" in violation of Section 4.1 pre-cooling requirements), and Heartland was 45 days past due on its July storage invoice ($18,200 unpaid!).
   - Unfavorable facts/documents (reveal honestly if student asks about unhelpful/adverse records or what really happened):
     (a) Internal Maintenance Ticket #409 & Slack messages from night shift supervisor: Bay C's secondary evaporator coil sensor WAS faulty on Aug 14–15, and a night technician silenced the alert for ~5.5 hours before moving the pallets, plus Marcus waited 36 hours to email Heartland while checking repair logs.""",
}


def list_sample_cases() -> List[Dict[str, Any]]:
    return [
        {
            "id": SAMPLE_NDIL_CONTRACT_CASE["id"],
            "title": SAMPLE_NDIL_CONTRACT_CASE["title"],
            "short_label": SAMPLE_NDIL_CONTRACT_CASE["short_label"],
            "jurisdiction": SAMPLE_NDIL_CONTRACT_CASE["jurisdiction"],
            "student_case_text": SAMPLE_NDIL_CONTRACT_CASE["student_case_text"],
        }
    ]


def get_case_bundle(case_id: Optional[str], custom_case_text: str = "", custom_client_notes: str = "") -> Dict[str, Any]:
    if case_id == SAMPLE_NDIL_CONTRACT_CASE["id"]:
        return {
            "case_id": SAMPLE_NDIL_CONTRACT_CASE["id"],
            "title": SAMPLE_NDIL_CONTRACT_CASE["title"],
            "student_case_text": custom_case_text.strip() or SAMPLE_NDIL_CONTRACT_CASE["student_case_text"],
            "client_simulation_profile": custom_client_notes.strip() or SAMPLE_NDIL_CONTRACT_CASE["client_simulation_profile"],
        }
    cleaned_case = custom_case_text.strip()
    cleaned_client = custom_client_notes.strip()
    return {
        "case_id": "custom" if cleaned_case else "none",
        "title": "Custom Case / Complaint" if cleaned_case else "Course Packet / Student-Provided Facts",
        "student_case_text": cleaned_case,
        "client_simulation_profile": cleaned_client,
    }

