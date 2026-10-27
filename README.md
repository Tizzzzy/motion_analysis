# Motion Mindset

A small local site to help new attorneys and law students think through *why*
they're filing a motion before they draft it, then practice working through
whether a complaint actually states a claim under Rule 12(b)(6).

The scope and design follow the law team's research document on legal
judgment prediction (Medvedeva & McBride) and their INCLUDE / LAW-TEAM REVIEW /
EXCLUDE framework for deciding which information a tool like this is allowed
to use.

## Pages

- `index.html` — overview of the tool, how it works, and why it reports
  pleading *completeness* instead of a predicted win likelihood.
- `mindset.html` — the Socratic / client-counseling walkthrough (12 guided
  questions across purpose, the judge's perspective, the opponent's
  perspective, consequences, and client counseling). This is the tool's answer
  to the document's noted gap: a way for students to submit their own
  reasoning, separate from the complaint text. Answers are saved in the
  browser's `localStorage`.
- `motion.html` — the Rule 12(b)(6) legal analysis page:
  - A collapsible panel makes the INCLUDE / LAW-TEAM REVIEW / EXCLUDE
    information categories explicit, including which "included" factors
    (court/district, other precedent, docket history) aren't actually wired
    up yet.
  - Claim type selector: only **Torts** has a built-out decision tree (duty,
    breach, causation, cognizable harm, then the *Twombly v. Iqbal*
    plausibility standard). Contracts and Property are honest stubs — the law
    team hasn't mapped those elements yet, so the tool doesn't guess at them.
  - For each Torts element, the student self-assesses Yes / Unsure / No and
    can note the supporting allegation. The result is a **pleading
    completeness** readout (e.g. "3 / 5 elements clearly supported") plus
    Socratic follow-up questions for any element marked unsure or missing —
    not an outcome prediction.
  - A stubbed "Feedback after you decide" section notes that the actual
    ruling is intentionally withheld from the initial analysis and reserved
    for comparison after the student has made their filing decision.

No backend is required yet — state lives in `localStorage`. The Torts
decision tree is a **labeled self-assessment scaffold**, not an AI judgment;
it stands in for the future LLM-backed version that would check the
student's answers against the complaint text itself.

## Run it

```bash
cd /gpfs/projects/p32143/ai_law_course
python3 -m http.server 8821
```

Then open `http://localhost:8821` in a browser. (Port 8000 is often already
in use by other users on shared HPC nodes — pick any free port.) If running
on a remote HPC/login node, forward the port over SSH:

```bash
ssh -L 8821:localhost:8821 <user>@<host>
```

Also deployed at https://motionanalysis.vercel.app/ (auto-deploys from the
`main` branch of this repo).

## Next steps (not yet built)

- Replace the self-assessment scaffold in `motion.html` with a real LLM call
  that reads the pasted complaint text directly, checks each element against
  it, and cites the specific allegation(s) it relied on.
- Build out the Contracts and Property decision trees once the law team
  defines their elements the same way they did for Torts.
- Add a way to capture the actual ruling after the fact and generate the
  "Feedback after you decide" comparison that's currently stubbed.
- Persist mindset answers + analysis sessions server-side per user/case
  instead of only in browser `localStorage`.
