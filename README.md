# Motion Mindset

A small local site to help new attorneys and law students think through *why*
they're filing a motion before they draft it, then practice getting
sentence-by-sentence feedback on the draft.

## Pages

- `index.html` — overview of the tool and how it works
- `mindset.html` — the mindset walkthrough (10 guided questions across purpose,
  the judge's perspective, the opponent's perspective, and consequences).
  Answers are saved in the browser's `localStorage`.
- `motion.html` — paste a motion draft and get sentence-by-sentence feedback
  plus an estimated likelihood of the motion being granted.

No backend is required yet — state lives in `localStorage`. The scoring on
`motion.html` is a **labeled placeholder heuristic** (keyword/length based),
standing in for the future LLM-backed analysis.

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

## Next steps (not yet built)

- Replace the heuristic in `motion.html`'s `scoreSentence()` with a real
  LLM call that takes the mindset answers + motion text and returns
  sentence-level reasoning and a calibrated acceptance probability.
- Persist mindset answers + motion drafts server-side per user/case instead
  of only in browser `localStorage`.
