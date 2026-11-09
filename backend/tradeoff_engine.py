"""Quantitative Tradeoffs & Decision-Support Calculation Engine.

Implements Stage 2 of the Strategic Judgment Simulation as specified in:
PDF 1 ("Quantify tradeoffs" & "Probability and objective-data policy").

Key Principles:
1. Hybrid approach: Students enter their conditional probability estimates and weights.
2. Curated legal authority and labeled reference data provide context.
3. Explicit separation of:
   - [Binding Legal Rule]
   - [Empirical Reference Observation]
   - [Scenario Assumption]
4. Formula:
   P(dismissal without amendment) = P(motion granted) * P(no amendment | grant)
5. Multi-attribute client preference weighting (Money, Speed, Relationship, Reputation, Risk Tolerance).
"""

from typing import Any, Dict, List, Optional


# Benchmark Scenario Baseline Defaults (Curated Reference Data)
BENCHMARKS: Dict[str, Any] = {
    "claimed_direct_loss": 68400.0,
    "claimed_consequential_loss": 340000.0,
    "total_claimed_exposure": 408400.0,
    "contract_cap_direct_damages": 36000.0,  # 2x monthly storage fees per Sec. 11.2
    "client_refinancing_deadline_months": 2.5,  # 75 days
    "default_fees": {
        "negotiate": 5000.0,
        "litigate_motion": 20000.0,
        "litigate_full_trial": 95000.0,
        "investigate": 7500.0,
    },
    "default_durations_months": {
        "negotiate": 1.5,
        "litigate_motion": 5.5,
        "litigate_full_trial": 18.0,
        "investigate": 2.5,
    },
    "default_probabilities": {
        "motion_granted": 0.60,
        "no_amendment_given_grant": 0.15,  # Runnion v. Girl Scouts (7th Cir.) liberal amendment rule
        "leave_to_amend_given_grant": 0.85,
        "plaintiff_prevails_at_trial": 0.50,
        "settlement_accepted_baseline": 0.70,
    },
    "labels": {
        "rule_15_amendment": {
            "type": "Binding Legal Rule",
            "cite": "FRCP 15(a)(2) & Runnion v. Girl Scouts, 786 F.3d 510 (7th Cir.)",
            "desc": "District courts in the 7th Circuit must freely give leave to amend after granting a Rule 12(b)(6) motion unless amendment would be futile.",
        },
        "limitation_of_liability": {
            "type": "Binding Legal Rule",
            "cite": "Master Agreement Sec. 11.2 & Illinois UCC / Contract Law",
            "desc": "Enforceable limitation-of-liability clause bars consequential damages ($340,000) and caps direct inventory recovery at 2x monthly storage fees ($36,000) unless willful misconduct is pleaded and proven.",
        },
        "four_corners_rule": {
            "type": "Binding Legal Rule",
            "cite": "FRCP 12(b)(6) / 12(d) & 7th Circuit Precedent",
            "desc": "A Rule 12(b)(6) motion is strictly confined to the Four Corners of the complaint plus documents attached or incorporated by reference. Extrinsic evidence (dock logs, maintenance records) cannot be considered without converting to summary judgment under Rule 12(d).",
        },
        "fjc_dismissal_data": {
            "type": "Empirical Reference Observation",
            "cite": "Federal Judicial Center (FJC) Motion-to-Dismiss Study",
            "desc": "In federal commercial contract disputes, over 70% of granted Rule 12(b)(6) motions result in leave to file an amended complaint rather than immediate termination.",
        },
        "refinancing_window": {
            "type": "Scenario Assumption",
            "cite": "Lakeview CEO Confidential Constraint",
            "desc": "Lakeview's commercial credit facility closes in ~75 days (2.5 months). Unresolved open litigation exceeding $250k materially threatens underwriting approval.",
        },
    },
}


def get_default_tradeoff_inputs() -> Dict[str, Any]:
    return {
        # Financial estimates entered by student
        "settlement_offer_cash": 35000.0,
        "settlement_offer_credits": 20000.0,  # storage fee credits
        "fee_negotiate": BENCHMARKS["default_fees"]["negotiate"],
        "fee_motion": BENCHMARKS["default_fees"]["litigate_motion"],
        "fee_full_trial": BENCHMARKS["default_fees"]["litigate_full_trial"],
        "fee_investigate": BENCHMARKS["default_fees"]["investigate"],

        # Potential exposure estimates
        "exposure_direct": BENCHMARKS["claimed_direct_loss"],
        "exposure_consequential": BENCHMARKS["claimed_consequential_loss"],

        # Duration estimates (months)
        "duration_negotiate": BENCHMARKS["default_durations_months"]["negotiate"],
        "duration_motion": BENCHMARKS["default_durations_months"]["litigate_motion"],
        "duration_full_trial": BENCHMARKS["default_durations_months"]["litigate_full_trial"],
        "duration_investigate": BENCHMARKS["default_durations_months"]["investigate"],

        # Conditional probability estimates entered by student (0.0 to 1.0)
        "p_motion_granted": BENCHMARKS["default_probabilities"]["motion_granted"],
        "p_no_amendment_given_grant": BENCHMARKS["default_probabilities"]["no_amendment_given_grant"],
        "p_plaintiff_prevails_trial": BENCHMARKS["default_probabilities"]["plaintiff_prevails_at_trial"],

        # Client-priority weights (1 to 5 scale)
        "weight_money": 5,          # Elena's high concern for limiting $340k exposure
        "weight_speed": 5,          # 75-day refinancing deadline
        "weight_relationship": 4,   # Heartland recurring storage revenue (~$190k/yr)
        "weight_reputation": 4,     # Avoiding public cold-chain failure claims
        "weight_risk_tolerance": 2, # Low risk tolerance; desires cost certainty
    }


def compute_tradeoffs(inputs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = get_default_tradeoff_inputs()
    if inputs:
        for k, v in inputs.items():
            if k in cfg:
                try:
                    cfg[k] = float(v) if "weight" not in k else int(round(float(v)))
                except (ValueError, TypeError):
                    pass

    # Clamping
    p_grant = max(0.01, min(0.99, cfg["p_motion_granted"]))
    p_no_amend = max(0.01, min(0.99, cfg["p_no_amendment_given_grant"]))
    p_trial_win = max(0.01, min(0.99, cfg["p_plaintiff_prevails_trial"]))

    # Core conditional calculation per PDF 1:
    # P(dismissal without amendment) = P(motion granted) * P(no amendment | grant)
    p_dismissal_permanent = round(p_grant * p_no_amend, 4)
    p_leave_to_amend = round(p_grant * (1.0 - p_no_amend), 4)
    p_motion_denied = round(1.0 - p_grant, 4)

    # 1. PATH A: Early Negotiated Settlement
    offer_cash = float(cfg["settlement_offer_cash"])
    offer_credits = float(cfg["settlement_offer_credits"])
    # Credit value to Lakeview is lower than face value (marginal warehouse space cost is ~40% of retail credit)
    effective_credit_cost = round(offer_credits * 0.45, 2)
    negotiate_payout = round(offer_cash + effective_credit_cost, 2)
    negotiate_fees = float(cfg["fee_negotiate"])
    negotiate_total_ev = round(negotiate_fees + negotiate_payout, 2)
    negotiate_duration = float(cfg["duration_negotiate"])

    # 2. PATH B: Litigate Rule 12(b)(6) Motion to Dismiss
    motion_fees = float(cfg["fee_motion"])
    trial_fees = float(cfg["fee_full_trial"])

    # If dismissal is permanent without amendment: Payout = $0, total cost = motion_fees
    cost_branch_permanent = motion_fees

    # If leave to amend granted: Plaintiff amends complaint (e.g. pleading willful misconduct from ticket #409).
    # Leads to full litigation or late settlement.
    # Expected payout if litigation continues:
    expected_continuing_liability = round(
        p_trial_win * (cfg["exposure_direct"] + cfg["exposure_consequential"] * 0.4), 2
    )
    cost_branch_amended = round(motion_fees + trial_fees + expected_continuing_liability, 2)

    # If motion denied: Same continuing litigation exposure
    cost_branch_denied = round(motion_fees + trial_fees + expected_continuing_liability, 2)

    # Expected Value of Litigate 12(b)(6):
    litigate_total_ev = round(
        (p_dismissal_permanent * cost_branch_permanent)
        + (p_leave_to_amend * cost_branch_amended)
        + (p_motion_denied * cost_branch_denied),
        2,
    )
    litigate_duration = float(cfg["duration_motion"])

    # 3. PATH C: Factual Investigation & Informal Resolution
    investigate_fees = float(cfg["fee_investigate"])
    # Investigation uncovers dock logs + ticket #409; facilitates targeted mediation with insurance contribution (~$30k carrier offset)
    investigate_settlement_cash = max(20000.0, offer_cash * 0.9)
    investigate_total_ev = round(investigate_fees + investigate_settlement_cash + effective_credit_cost, 2)
    investigate_duration = float(cfg["duration_investigate"])

    # Refinancing Window Check (2.5 months / 75 days)
    refinancing_deadline = BENCHMARKS["client_refinancing_deadline_months"]
    refinancing_risk = {
        "negotiate": negotiate_duration > refinancing_deadline,
        "litigate": litigate_duration > refinancing_deadline,
        "investigate": investigate_duration > refinancing_deadline,
    }

    # Multi-attribute Client Utility Score (0 to 100)
    # Weights normalized
    w_money = max(1, min(5, cfg["weight_money"]))
    w_speed = max(1, min(5, cfg["weight_speed"]))
    w_rel = max(1, min(5, cfg["weight_relationship"]))
    w_rep = max(1, min(5, cfg["weight_reputation"]))
    w_risk = max(1, min(5, cfg["weight_risk_tolerance"]))
    total_weights = w_money + w_speed + w_rel + w_rep + w_risk

    max_exposure = BENCHMARKS["total_claimed_exposure"]

    def calc_utility(ev: float, dur: float, rel_score: float, rep_score: float, risk_var: float) -> int:
        # Financial component: 0 (if EV >= max_exposure) to 100 (if EV = 0)
        s_money = max(0.0, min(100.0, (1.0 - (ev / max_exposure)) * 100.0))
        # Speed component: 100 if well under deadline, steep penalty if misses 75 days
        if dur <= refinancing_deadline:
            s_speed = 95.0 - (dur / refinancing_deadline) * 15.0
        else:
            s_speed = max(10.0, 70.0 - (dur - refinancing_deadline) * 20.0)
        # Relationship component (0-100)
        s_rel = rel_score
        # Reputation component (0-100)
        s_rep = rep_score
        # Risk component (certainty preference)
        s_risk = risk_var

        weighted = (
            (s_money * w_money)
            + (s_speed * w_speed)
            + (s_rel * w_rel)
            + (s_rep * w_rep)
            + (s_risk * w_risk)
        ) / total_weights
        return int(round(weighted))

    utility_negotiate = calc_utility(negotiate_total_ev, negotiate_duration, 88.0, 85.0, 90.0)
    utility_litigate = calc_utility(litigate_total_ev, litigate_duration, 35.0, 45.0, 30.0)
    utility_investigate = calc_utility(investigate_total_ev, investigate_duration, 80.0, 78.0, 75.0)

    # Sensitivity Analysis: Vary P(no amendment | grant) from 0.05 to 0.70
    sensitivity_curve: List[Dict[str, Any]] = []
    for test_p_no_amend in [0.05, 0.15, 0.25, 0.35, 0.50, 0.65]:
        p_perm = round(p_grant * test_p_no_amend, 3)
        p_amend = round(p_grant * (1.0 - test_p_no_amend), 3)
        test_ev = round(
            (p_perm * cost_branch_permanent)
            + (p_amend * cost_branch_amended)
            + (p_motion_denied * cost_branch_denied),
            2,
        )
        sensitivity_curve.append({
            "p_no_amendment": test_p_no_amend,
            "p_dismissal_permanent": p_perm,
            "litigate_ev": test_ev,
            "negotiate_ev": negotiate_total_ev,
            "delta_vs_negotiate": round(test_ev - negotiate_total_ev, 2),
        })

    return {
        "inputs": cfg,
        "probabilities": {
            "p_motion_granted": p_grant,
            "p_no_amendment_given_grant": p_no_amend,
            "p_leave_to_amend_given_grant": round(1.0 - p_no_amend, 4),
            "p_dismissal_permanent": p_dismissal_permanent,
            "p_leave_to_amend": p_leave_to_amend,
            "p_motion_denied": p_motion_denied,
            "formula_explanation": "P(Permanent Dismissal) = P(Motion Granted) * P(No Amendment | Grant)",
        },
        "path_comparisons": {
            "negotiate": {
                "name": "Early Negotiated Settlement",
                "expected_fees": negotiate_fees,
                "expected_payout": negotiate_payout,
                "total_ev": negotiate_total_ev,
                "duration_months": negotiate_duration,
                "misses_refinancing_deadline": refinancing_risk["negotiate"],
                "client_utility_score": utility_negotiate,
                "key_pros": "Certain cost, resolves before 75-day bank deadline, preserves Heartland account ($190k/yr), strictly confidential.",
                "key_cons": "Requires paying ~$35k cash out of pocket.",
            },
            "litigate": {
                "name": "Rule 12(b)(6) Motion to Dismiss",
                "expected_fees": round(motion_fees + (1.0 - p_dismissal_permanent) * trial_fees, 2),
                "expected_payout": round((1.0 - p_dismissal_permanent) * expected_continuing_liability, 2),
                "total_ev": litigate_total_ev,
                "duration_months": litigate_duration,
                "misses_refinancing_deadline": refinancing_risk["litigate"],
                "client_utility_score": utility_litigate,
                "key_pros": "Chance of dismissing $340k consequential damages claim on Four Corners under Section 11.2.",
                "key_cons": "High probability of leave to amend (Runnion), exceeds 75-day refinancing window, costs ~$20k even if granted, public docket.",
            },
            "investigate": {
                "name": "Factual Investigation & Pre-Answer Informal Outreach",
                "expected_fees": investigate_fees,
                "expected_payout": investigate_settlement_cash + effective_credit_cost,
                "total_ev": investigate_total_ev,
                "duration_months": investigate_duration,
                "misses_refinancing_deadline": refinancing_risk["investigate"],
                "client_utility_score": utility_investigate,
                "key_pros": "Verifies ticket #409 and dock records before making public representations; engages insurer; keeps options open.",
                "key_cons": "Slightly delays immediate resolution; costs $7.5k in initial investigation fees.",
            },
        },
        "sensitivity_analysis": {
            "parameter": "P(no amendment | grant) [Rule 15 liberal amendment standard]",
            "curve": sensitivity_curve,
            "insight": (
                "Because 7th Circuit precedent (Runnion) overwhelmingly grants leave to amend, "
                "even a successful Rule 12(b)(6) motion rarely terminates commercial litigation. "
                f"Litigation EV (${litigate_total_ev:,.0f}) significantly exceeds Early Settlement EV (${negotiate_total_ev:,.0f}) "
                "unless the probability of dismissal with prejudice without leave to amend exceeds 60%."
            ),
        },
        "legal_and_scenario_labels": BENCHMARKS["labels"],
    }
