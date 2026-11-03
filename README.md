# Motion Mindset — Conversational Motion Guidance

A client-centered, conversational training site for law students and junior attorneys preparing to counsel a client and evaluate/draft a motion (currently centered on Illinois breach-of-contract claims and the Seventh Circuit Rule 12(b)(6) framework).

## Architecture & Pages

- **`index.html` — Home Overview**: Introduces the three-phase simulation, displays live Question Bank and Evaluation Rubric counts, and explains why the system evaluates legal/counseling completeness rather than judicial outcome probabilities (Medvedeva & McBride).
- **`guidance.html` — Conversational Guidance**:
  1. **Before Meeting (`Mentor Q&A`)**: Begins immediately with *"Who is the client and who can authorize decisions?"* Your Supervising Partner evaluates each response against the CSV rubric (`0–3`), coaches revisions Socratically when below competent (`< 2`), and advances automatically once competent (`2` or `3`).
  2. **During Meeting (`Client Roleplay`)**: Roles switch — you interview the simulated client directly to uncover ranked priorities, budget limits, settlement conditions, business relationships, deadlines, risk preferences, supporting/adverse facts, and formal approval authority while the rubric evaluator tracks your coverage and rapport.
  3. **After Meeting (`Mentor Debrief`)**: Synthesize client priorities, analyze pleading plausibility under Illinois contract law (*Zahdan*, 735 ILCS 5/2-603) and the 7th Circuit Rule 12(b)(6) *Twombly/Iqbal* framework (Four Corners & Incorporation-by-Reference), weigh alternatives, and formulate a client-aligned recommendation.
  - Includes an **optional Case File side panel** with a one-click **Sample NDIL Breach-of-Contract Case** (*Heartland Specialty Foods v. Lakeview Cold Chain Logistics*) or custom pasted complaint text, plus live **Initial & Revised Rubric Scoring** and CSV export.
- **`motion.html` — Motion Drafting Studio**: Carries your conversational session insights and complaint text into a structured editor (full Rule 12(b)(6) motion, partial motion, or strategic counseling memo) with live Qwen 3.8-27B Partner Review.

## Dynamic CSV Hot-Reloading

Both curriculum spreadsheets are read dynamically and watched by modification timestamp (`mtime`):

- `Client Simulation Question Bank(Question Bank).csv`
- `Client Simulation Question Bank(Evaluation Rubric).csv`

Replacing or updating either CSV file in the repository automatically updates phases, questions, evaluation criteria, extra columns, and rubric skills **without any code changes** (including UTF-8 and Windows-1252 encodings and skipping blank placeholder rows).

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

What `run_qwen_with_tunnel.sh` does automatically:
1. Generates and persists a secret Bearer token in `data/qwen_api_key.txt` (`QWEN_API_KEY`) so only your Vercel backend can call your GPU.
2. Activates `qwen` + `module load vllm/0.19.0` and starts `vllm_server.py` on port `8001`.
3. Downloads standalone `cloudflared` to `bin/cloudflared` (if not already present), opens an outbound HTTP/2 TLS tunnel, and prints:
   - `VLLM_URL` (e.g. `https://xxxx.trycloudflare.com`)
   - `QWEN_API_KEY` (e.g. `qwen_sk_...`)

> **Tip — Permanent Tunnel URL across SLURM Jobs**:
> Quick Tunnel URLs (`*.trycloudflare.com`) change when a SLURM job restarts. To keep `VLLM_URL` fixed permanently so you never have to touch Vercel settings again, pass either a free static ngrok domain (`NGROK_DOMAIN=your-name.ngrok-free.app ./run_qwen_with_tunnel.sh`) or a Cloudflare Named Tunnel token (`CLOUDFLARE_TUNNEL_TOKEN=... ./run_qwen_with_tunnel.sh`).

### 2. Configure Vercel Environment Variables & Deploy

1. In your **Vercel Project Dashboard** (`Settings` → `Environment Variables`), add:
   - `VLLM_URL` = the HTTPS tunnel URL printed by `run_qwen_with_tunnel.sh`
   - `QWEN_API_KEY` = the key printed by `run_qwen_with_tunnel.sh` (from `data/qwen_api_key.txt`)
   - *(Note: Ensure your Vercel project's Function Max Duration supports 30–60s; `vercel.json` requests `maxDuration: 60`.)*
2. Commit and push to `main` (or run `vercel --prod`) to deploy. When updating the CSV files later, simply replace the CSVs and push to `main`.

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
./run_web.sh --port 8821 --vllm-url http://qgpu2008:8001
```

*(Or run everything inside a single process directly on the GPU node via `python server.py --port 8821 --local-vllm`.)*

Then open `http://localhost:8821` in your browser.
