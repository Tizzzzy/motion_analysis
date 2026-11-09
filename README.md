# Innovation Lab | Strategic Judgment Simulation

A controlled, interactive Illinois breach-of-contract hypothetical that develops law students' and junior attorneys' strategic judgment.

> **Working Direction (from Project Specification):**
> *"Build a controlled, interactive Illinois breach-of-contract hypothetical that develops students' strategic judgment. The project is not primarily a motion-outcome predictor, case-upload analyzer, or litigation-procedure tutorial. Students should learn to discover client priorities, compare legally defensible strategies, anticipate opposing-party reactions, quantify uncertainty, and explain or revise a recommendation. A strong legal argument may still be a poor strategic choice for a particular client."*

---

## The 6-Stage Student Experience & Branching Model

1. **Stage 1 &middot; Client Interview**:
   - Ask open-ended questions to simulated client representatives (**Elena Rostova**, CEO, and **Marcus Vance**, Warehouse Operations).
   - Discover hidden corporate authority thresholds (CEO authority up to \$25,000; CFO sign-off required above \$25,000; warehouse ops has no dispositive filing authority).
   - Elicit and rank the client's **Top Three Priorities** (limiting \$340k exposure, protecting cold-chain reputation, and resolving before the **75-day commercial bank refinancing deadline**).
2. **Stage 2 &middot; Quantify Tradeoffs (Quantitative Decision Support)**:
   - Interactive calculation engine for legal costs, liabilities, resolution times, and conditional probabilities.
   - Core 7th Circuit Formula: `P(dismissal without amendment) = P(motion granted) * P(no amendment | grant)`.
   - Explicit pedagogical separation of:
     - `[Binding Legal Rule]` (e.g. FRCP 15(a)(2) / *Runnion* liberal amendment rule; Master Agreement Sec. 11.2 limitation of liability).
     - `[Empirical Reference Observation]` (e.g. FJC motion-to-dismiss empirical rates).
     - `[Scenario Assumption]` (e.g. 75-day bank credit refinancing window).
   - Real-time Expected Value (EV), multi-attribute client utility scores (0–100%), and sensitivity analysis curves.
3. **Stage 3 &middot; Opponent Anticipation**:
   - Analyze opposing party (Heartland Specialty Foods) commercial pressures (dairy supplier debt, retail placement, spoilage loss).
   - Anticipate opposing counsel's reaction to settlement versus a Rule 12(b)(6) motion, with instant Socratic feedback.
4. **Stage 4 &middot; Initial Recommendation (Strategy Selection)**:
   - Commit to a defensible course of action:
     - **Option A &middot; Structured Early Settlement** (\$35k–\$42.5k cash + storage credits + mutual release + confidentiality).
     - **Option B &middot; Rule 12(b)(6) Motion to Dismiss** (targeting Count I consequential damages under Sec. 11.2).
     - **Option C &middot; Pre-Answer Factual Investigation** (informal 30-day extension, sensor telemetry audit, insurance notice).
   - Record decision and detailed strategic rationale tied to client business constraints.
5. **Stage 5 &middot; Experience a Simulated Event**:
   - Scripted, controlled scenario event triggered based on the student's strategy:
     - *Litigate*: Judge Kendall issues order dismissing consequential damages without prejudice, but grants 21 days leave to amend under *Runnion*. Heartland amends pleading willful misconduct via leaked maintenance ticket #409, spending \$19,400 in fees and **missing the 75-day bank loan deadline**.
     - *Negotiate*: Heartland counters with \$42,500 cash + \$20k storage credits, settling within 38 days before bank review.
     - *Investigate*: Audit uncovers dock logs + ticket #409, secures \$30k insurance contribution and 45-day mediation standstill.
6. **Stage 6 &middot; Reflect, Revise & 100-Point Faculty Rubric Assessment**:
   - Pre/post comparison of reasoning before and after experiencing the event.
   - Automated Socratic grading against the official **100-Point Faculty Rubric**:
     1. Client Interviewing (25 pts)
     2. Legal Reasoning (25 pts)
     3. Risk & Uncertainty (20 pts)
     4. Strategic Judgment (20 pts)
     5. Reflection & Revision (10 pts)
   - Performance anchors: **Strong** (90–100), **Competent** (75–89), **Developing** (60–74), **Needs Improvement** (<60).
   - Downloadable CSV scorecard report and printable PDF assessment.

---

## Architecture & Workspaces

- **`index.html` — Overview**: Homepage introducing the Innovation Lab simulation, learning goals, case background, and 100-point rubric breakdown.
- **`simulation.html` — 6-Stage Simulation Workspace**: The full interactive simulation application covering Stages 1 through 6.
- **`motion.html` — Motion Drafting Studio**: Editor for drafting Rule 12(b)(6) briefs or client counseling memos, carrying over simulation findings and case notes.
- **`guidance.html` & `mindset.html`**: Automatically route to `simulation.html`.

---

## Option A: Deploy Public Website on Vercel + Run Qwen 3.8-27B on Quest GPU

In this hybrid architecture, **Vercel** hosts the public website and Python `/api/*` serverless routes (statelessly syncing session snapshots with the browser), while **Northwestern Quest** runs **Qwen 3.8-27B** on GPU and exposes an outbound Bearer-authenticated HTTPS tunnel.

### 1. Start Qwen 3.8-27B + HTTPS Tunnel on Quest

Inside an active Quest GPU allocation (2 GPUs recommended for `tensor_parallel_size=2`):

```bash
cd /projects/p32143/ai_law_course
./run_qwen_with_tunnel.sh
```

Or launch directly from a Quest login node into an active SLURM GPU job (`squeue -u $USER`):

```bash
srun --jobid <JOBID> --overlap /projects/p32143/ai_law_course/run_qwen_with_tunnel.sh
```

`run_qwen_with_tunnel.sh` automatically:
1. Generates and persists a secret Bearer token in `data/qwen_api_key.txt` (`QWEN_API_KEY`) so only your Vercel backend can call your GPU.
2. Activates `qwen` + `module load vllm/0.19.0` and starts `vllm_server.py` on port `8001`.
3. Downloads standalone `cloudflared` to `bin/cloudflared` (if not already present), opens an outbound HTTP/2 TLS tunnel, and prints:
   - `VLLM_URL` (e.g. `https://xxxx.trycloudflare.com`)
   - `QWEN_API_KEY` (e.g. `qwen_sk_...`)

### 2. Configure Vercel Environment Variables & Deploy

1. In your **Vercel Project Dashboard** (`Settings` → `Environment Variables`), add:
   - `VLLM_URL` = the HTTPS tunnel URL printed by `run_qwen_with_tunnel.sh`
   - `QWEN_API_KEY` = the key printed by `run_qwen_with_tunnel.sh` (from `data/qwen_api_key.txt`)
2. Commit and push to `main` (or run `vercel --prod`) to deploy.

---

## Option B: Run Everything Locally on Quest (`localhost:8821`)

### Step 1: Start the Qwen 3.8-27B vLLM Worker on a GPU Node

```bash
cd /projects/p32143/ai_law_course
./run_qwen_server.sh --port 8001
```

### Step 2: Start the Web & API Server

On your login/IDE node (e.g., `quser42` for automatic VS Code port forwarding) or on the GPU node:

```bash
cd /projects/p32143/ai_law_course
./run_web.sh --port 8821 --vllm-url http://qgpu2009:8001
```

Then open `http://localhost:8821` in your browser.
