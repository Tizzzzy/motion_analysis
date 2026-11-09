"""Simulated Event Engine (Branching Event Model).

Implements Stage 5 of the Strategic Judgment Simulation as specified in:
PDF 1 ("Experience an event" & "Event vs. analysis: implementation distinction")
PDF 2 ("Case Background & Confidential Client Priorities")
PDF 3 ("Intake Note & Complaint").

CRITICAL PEDAGOGICAL POLICY:
- Trigger scripted / controlled opponent or court event grounded in the scenario.
- Student conditional probability estimates do NOT alter or control the actual simulated event.
- Exposes the practical limits of reflexive motion practice (Rule 15 liberal amendment in the 7th Cir.)
  and the commercial impact on client business deadlines (75-day bank refinancing).
"""

from typing import Any, Dict, Optional


def trigger_simulated_event(strategy_choice: str, strategy_details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    strat = (strategy_choice or "negotiate").strip().lower()
    details = strategy_details or {}

    if "litigat" in strat or "motion" in strat or "12(b)(6)" in strat:
        return _event_litigate_motion(details)
    elif "investigat" in strat or "inquiry" in strat or "pre_answer" in strat:
        return _event_investigate(details)
    else:
        return _event_negotiate(details)


def _event_litigate_motion(details: Dict[str, Any]) -> Dict[str, Any]:
    motion_scope = details.get("motion_scope", "partial_dismissal_consequential")
    return {
        "event_id": "event_court_ruling_runnion",
        "branch": "litigate",
        "title": "U.S. District Court Ruling on Defendant's Rule 12(b)(6) Motion to Dismiss",
        "court": "U.S. District Court for the Northern District of Illinois (Eastern Division)",
        "judge": "Hon. Virginia M. Kendall, U.S. District Judge",
        "date_elapsed_days": 112,  # ~3.7 months post-complaint
        "fees_incurred": 19400.0,
        "refinancing_impact": {
            "deadline_days": 75,
            "elapsed_days": 112,
            "missed_deadline": True,
            "bank_status": "CRITICAL HOLD — Bank underwriting paused Lakeview's $2.8M credit facility refinancing due to open, unresolved federal litigation exceeding $250,000 threshold.",
        },
        "court_order_text": (
            "MEMORANDUM OPINION AND ORDER\n\n"
            "Defendant Lakeview Cold Chain Logistics, LLC moves under Federal Rule of Civil Procedure 12(b)(6) "
            "to dismiss Count I of Plaintiff Heartland Specialty Foods, Inc.'s Complaint, or in the alternative, "
            "to dismiss Plaintiff's demand for $340,000 in consequential damages.\n\n"
            "1. Four Corners & Incorporation by Reference: Under Seventh Circuit precedent (188 LLC v. Trinity Indus., Inc., "
            "300 F.3d 730), the March 15, 2024 Master Agreement is referenced in the Complaint and central to Plaintiff's claim. "
            "Section 11.2 contains a plain, conspicuous waiver of consequential, lost-profit, and reputational damages.\n\n"
            "2. Dismissal in Part: Because the Complaint as pleaded alleges only ordinary breach of contract without well-pleaded "
            "factual allegations of intentional or reckless wrongdoing, Plaintiff's claim for $340,000 in consequential damages is DISMISSED.\n\n"
            "3. Leave to Amend Granted (FRCP 15(a)(2)): In this Circuit, 'a plaintiff whose original complaint has been dismissed "
            "under Rule 12(b)(6) should be given at least one opportunity to try to amend her complaint before the entire action is dismissed.' "
            "Runnion v. Girl Scouts of Greater Chi. & Nw. Ind., 786 F.3d 510, 519 (7th Cir. 2015). Plaintiff is granted 21 days leave to file an Amended Complaint."
        ),
        "opponent_reaction": {
            "counsel": "Heartland Litigation Counsel (Miller & Vance, LLP)",
            "action_taken": "Filed First Amended Complaint within 18 days.",
            "new_factual_allegation": (
                "Heartland's Amended Complaint attaches Lakeview Internal Maintenance Ticket #409 (obtained from a former "
                "warehouse technician). It alleges that Bay C's secondary evaporator alarm sounded at 02:15 AM on August 15, "
                "and that Lakeview staff intentionally silenced the alarm for 5.5 hours to avoid waking supervisors, thereby "
                "constituting 'willful and wanton misconduct' that invalidates the Section 11.2 limitation of liability."
            ),
        },
        "client_executive_reaction": {
            "speaker": "Elena Rostova, CEO",
            "quote": (
                "Our bank underwriters just flagged this lawsuit today and delayed our credit-facility refinancing! "
                "We spent nearly $20,000 on this motion, yet the case is still wide open, Heartland is now claiming "
                "willful misconduct on the public record, and we're in Month 4. Marcus told me this motion would get the case "
                "thrown out in 30 days. Why didn't our legal strategy anticipate that Heartland would just amend?"
            ),
        },
        "learning_takeaway": (
            "A Rule 12(b)(6) motion is rarely a final exit ramp in federal practice because Rule 15 and 7th Circuit law "
            "(Runnion) liberally grant leave to amend. Reflexive motion practice incurred $19k in fees, invited an amended complaint "
            "alleging intentional misconduct, and blew past Lakeview's critical 75-day bank refinancing deadline."
        ),
    }


def _event_negotiate(details: Dict[str, Any]) -> Dict[str, Any]:
    cash_offer = float(details.get("settlement_cash", 35000.0) or 0)
    credits_offer = float(details.get("storage_credits", 20000.0) or 0)
    total_package = cash_offer + credits_offer
    includes_confidentiality = bool(details.get("confidentiality", True))
    no_admission = bool(details.get("no_admission", True))

    if cash_offer >= 30000.0 and total_package >= 50000.0 and includes_confidentiality and no_admission:
        # Realistic Strong Offer -> Opponent Acceptance / Modest Counter
        return {
            "event_id": "event_negotiate_success",
            "branch": "negotiate",
            "title": "Heartland Leadership Responds to Confidential Settlement Offer",
            "date_elapsed_days": 38,  # ~1.2 months
            "fees_incurred": 4800.0,
            "refinancing_impact": {
                "deadline_days": 75,
                "elapsed_days": 38,
                "missed_deadline": False,
                "bank_status": "CLEARED — Lawsuit resolved and dismissed with prejudice 37 days before bank refinancing review.",
            },
            "opponent_letter": (
                "CONFIDENTIAL SETTLEMENT COMMUNICATION (FRE 408 / IRE 408)\n\n"
                "Dear Counsel:\n"
                "Heartland Specialty Foods, Inc. has reviewed Lakeview's structured settlement proposal. "
                "While Heartland maintains that its direct inventory and retail displacement losses exceeded $400,000, "
                "our executive team values the commercial relationship and recognizes the utility of future warehousing capacity.\n\n"
                "Heartland is prepared to accept the following counter-terms for a complete, final resolution:\n"
                "1. Cash Payment: $42,500 payable within 14 days (Lakeview's CEO and CFO approve this within their combined authority).\n"
                "2. Storage Credit: $25,000 in fee credits applied evenly over the next 12 months at Bolingbrook Bay C.\n"
                "3. Mutual General Release of all claims relating to the August 14-16, 2026 spoilage.\n"
                "4. Strict Non-Disclosure & Non-Disparagement agreement protecting both companies' reputations.\n"
                "5. Stipulation of Dismissal With Prejudice to be filed in the Northern District of Illinois within 5 business days."
            ),
            "client_executive_reaction": {
                "speaker": "Elena Rostova, CEO & David Park, CFO",
                "quote": (
                    "This is an outstanding outcome. By combining $42.5k cash with warehouse credits, we resolved a $408,000 "
                    "lawsuit for less than $50k total outlay, our legal fees were under $5,000, and our bank credit line closes "
                    "next month with zero litigation red flags. Best of all, Heartland remains our customer for 2027."
                ),
            },
            "learning_takeaway": (
                "Aligning legal strategy with client business priorities (preserving a $190k/yr commercial relationship, "
                "protecting the 75-day refinancing deadline, and using non-monetary storage credits) achieved a superior "
                "result compared to expensive, prolonged dispositive motion practice."
            ),
        }
    elif cash_offer >= 15000.0:
        # Moderate / Stiff Counter
        return {
            "event_id": "event_negotiate_counter",
            "branch": "negotiate",
            "title": "Heartland Rejects Low Cash Offer; Demands Cash Security and Audit Rights",
            "date_elapsed_days": 44,
            "fees_incurred": 6500.0,
            "refinancing_impact": {
                "deadline_days": 75,
                "elapsed_days": 44,
                "missed_deadline": False,
                "bank_status": "PENDING — Negotiation in progress; 31 days remaining before bank review deadline.",
            },
            "opponent_letter": (
                "Heartland rejects Lakeview's offer as insufficient to cover wholesale spoilage. "
                "Heartland counters with a demand for $55,000 in immediate cash, plus independent third-party temperature "
                "monitoring audits of Bay C at Lakeview's expense. Heartland gives Lakeview 10 days to accept before "
                "scheduling depositions of Marcus Vance and CEO Elena Rostova."
            ),
            "client_executive_reaction": {
                "speaker": "Elena Rostova, CEO",
                "quote": (
                    "We need to decide whether to bridge the gap with another $10,000 cash or bring in our insurance carrier. "
                    "At least we are talking directly rather than spending $20,000 on federal court briefs."
                ),
            },
            "learning_takeaway": (
                "Commercial negotiation involves anchored counter-proposals. The student has preserved the refinancing window "
                "and can now evaluate whether to bridge the remaining financial gap."
            ),
        }
    else:
        # Lowball offer (<$15k)
        return {
            "event_id": "event_negotiate_rejected",
            "branch": "negotiate",
            "title": "Heartland Counsel Rejects Lowball Offer and Escalates Discovery",
            "date_elapsed_days": 35,
            "fees_incurred": 7200.0,
            "refinancing_impact": {
                "deadline_days": 75,
                "elapsed_days": 35,
                "missed_deadline": False,
                "bank_status": "AT RISK — Discovery demands and hostile tone signal extended federal litigation.",
            },
            "opponent_letter": (
                "Heartland's counsel writes: 'Your offer does not even cover half of Heartland's out-of-pocket wholesale cheese loss ($68,400). "
                "We view this as bad-faith posturing. Heartland has noticed Marcus Vance's deposition for the 15th and served "
                "Rule 34 document requests demanding all compressor sensor logs for 2025–2026.'"
            ),
            "client_executive_reaction": {
                "speaker": "Elena Rostova, CEO",
                "quote": (
                    "A $10,000 offer was clearly too low for an inventory loss of $68k. Now Marcus is subpoenaed for a deposition "
                    "and our warehouse supervisors will be distracted right before the holiday rush."
                ),
            },
            "learning_takeaway": (
                "Offers that fall well below plausible out-of-pocket losses often trigger retaliatory escalation. "
                "Effective negotiation requires grounding proposals in the opponent's legitimate baseline direct damages."
            ),
        }


def _event_investigate(details: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "event_id": "event_investigation_findings",
        "branch": "investigate",
        "title": "Pre-Answer Factual Audit Findings & Rule 408 Counsel Conference",
        "date_elapsed_days": 32,  # ~1 month
        "fees_incurred": 6800.0,
        "refinancing_impact": {
            "deadline_days": 75,
            "elapsed_days": 32,
            "missed_deadline": False,
            "bank_status": "CONTROLLED — 45-day standstill agreement keeps docket clean during bank underwriting review.",
        },
        "audit_findings_text": (
            "INTERNAL INVESTIGATION & RECORD AUDIT SUMMARY:\n\n"
            "1. Receiving Dock Logs (Aug 12, 2026): Pallets 1 and 2 arrived from Heartland's dairy carrier at 45.2°F, "
            "violating Section 4.1's pre-cooling requirement. Furthermore, Heartland was 45 days past due on its July invoice ($18,200 unpaid).\n\n"
            "2. Maintenance Ticket #409 (Aug 14-15, 2026): A night technician silenced Bay C's secondary evaporator alarm "
            "for 5.5 hours, and Marcus Vance waited 36 hours before alerting Heartland. This fact is fatal if exposed in open litigation.\n\n"
            "3. Insurance Position: Lakeview's general liability carrier confirms a $50,000 warehouseman's legal liability "
            "sub-limit under a reservation of rights, and agrees to contribute $30,000 toward an early mediated settlement.\n\n"
            "4. Informal Rule 408 Conference with Heartland: Sharing the dock log and past-due invoice off the record "
            "persuaded Heartland's counsel to pause federal deadlines. The parties entered an agreed 45-day standstill and "
            "agreed to attend mediation before former Magistrate Judge Schenkier."
        ),
        "client_executive_reaction": {
            "speaker": "Elena Rostova, CEO",
            "quote": (
                "Digging into our warehouse records before filing a motion saved us from a catastrophe. If we had filed "
                "Marcus's motion, Heartland would have discovered Ticket #409 in discovery and destroyed our reputation. "
                "Now we have our insurance carrier contributing $30k and Heartland agreeing to mediation before our bank review."
            ),
        },
        "learning_takeaway": (
            "Investigating facts before taking a public legal position is sound strategic counseling (Model Rule 1.1 / 1.4). "
            "Uncovering both supporting facts (dock logs) and unfavorable vulnerabilities (ticket #409) allowed counsel to "
            "broker an insurance-backed mediation within the client's critical 75-day refinancing window."
        ),
    }
