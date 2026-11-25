# Telecom Customer Churn Insight Assistant

An AI-assisted customer retention and churn-insight service built with **Python**, **LangGraph**, **Ollama (`gpt-oss:120b-cloud`)**, **FastAPI**, and **Streamlit**.

It summarizes verified customer tickets, calls, billing disputes, and usage changes, classifies churn risk deterministically in code, recommends one within-policy retention action from a fixed catalog, and escalates policy exceptions, unresolved conflicts, or citation failures to a human supervisor via LangGraph `interrupt()`.

---

## Architecture (8-Node LangGraph Flow, 1 Escalation Decision)

```
Request ──► Check input ──► Retrieve records ──► Score risk ──► Generate ──► Guard checks
                                                                                 │
                                                                           <Escalate?>
                                                                            ├── yes ──► Supervisor review (interrupt) ──┐
                                                                            └── no ─────────────────────────────────────┴──► Response
```

---

## Prompt Design, Data Assumptions & Governance (Required Answers)

### 1. Which record wins on conflict and why?
When multiple verified records share the same `dispute_group` with contradictory statuses (for example, `CUST-1002` where earlier ticket `TKT-6610` on `2026-10-02` says `resolved`, while later call `CALL-8821` on `2026-10-12` reports `open_dispute` because the Rs 1,240 roaming double charge is still on the bill), **the latest verified record by date (`CALL-8821`) wins**.  
**Why:** Customer service state evolves over time; a subsequent customer call reporting that a supposedly resolved charge still appears on the live bill supersedes the earlier ticket closure. Both conflicting record IDs (`TKT-6610` and `CALL-8821`) are explicitly surfaced in `data_conflicts` so the agent sees the contradiction. If two contradictory verified records carry the **exact same date** and cannot be ordered by recency, `has_unresolved_conflict` is set to `True` and the request escalates to a supervisor.

### 2. What counts as "verified"?
A record or policy counts as verified **only when its metadata carries `"verified": true`**.
- Draft agent notes or scratchpad entries with `"verified": false` (such as `NOTE-77` on `CUST-1001`) are excluded during retrieval, never passed to risk scoring, and blocked by `Guard checks` if a model ever attempts to cite them.
- Policy documents must also be in-date for the request's `session_date` (`valid_from <= session_date <= valid_to`); expired offers such as `OFFER-FEST` (`2026-09-25` to `2026-10-08` when `session_date` is `2026-10-15`) are never active or recommended.

### 3. How is the risk score computed and why can the model not override it?
Risk is computed **deterministically in Python** (`src/churn_assistant/scoring.py`) before the generation node runs, using fixed weights:
- `network_complaints_3plus` (3+ verified network complaints from the same cell within 30 days): **+3**
- `open_disputed_billing` (latest verified billing record still open/disputed): **+3**
- `competitor_or_port_out` (competitor mention or port-out request): **+3**
- `usage_drop_30pct` (data usage down >= 30% month on month): **+2**
- `repeat_contact` (2+ verified interactions on the same issue/location/group): **+1**

Fixed band mapping: **0–2 = `low`**, **3–5 = `medium`**, **6+ = `high`** (`null` when no verified interaction/usage signals exist, in which case `missing_information` must be non-empty).  
**Why the model cannot override it:** Churn risk drives SLA priority and escalation routing. Keeping scoring in deterministic code prevents prompt sensitivity, tone bias, or unverified text from inflating or deflating risk, and `Guard checks` overwrite `risk` with the code-computed `RiskAssessment` even if the LLM attempts to alter it.

### 4. Which requests are always escalated?
Guard checks (never the LLM) set `escalate: true` and pause the graph at `Supervisor review` (`interrupt()`) whenever:
1. **Policy exception:** Requested or needed retention offer exceeds `RET-POL` limits (bill credit > Rs 500 or plan discount > 20% or > 3 months—such as `CUST-1004` asking for 50% off for 6 months).
2. **Unresolved data conflict:** Contradictory verified records cannot be resolved by recency.
3. **Failed citation / verification check:** A draft response cites an unknown record ID, cites a `verified: false` record, or proposes an action without a valid scored signal.

### 5. What can the assistant not know?
- **Unknown customers:** Any `customer_id` absent from the customer directory (e.g., `CUST-9999`) is turned back at `Check input` with no invented history.
- **Competitor pricing:** No record contains competitor tariffs or pricing tables, so price comparisons are always reported under `unsupported`.
- **Network repair dates / ETAs:** Engineering restoration schedules are not in approved CRM/policy records, so repair dates are always reported under `unsupported`.
- **Unrecorded real-time customer context:** Preferences such as the customer's preferred callback time or missing recent interactions (`CUST-1005`) are listed under `missing_information` rather than guessed.
- **Personal Identifiable Information (PII):** Phone numbers, addresses, emails, national IDs, and card numbers are masked at `Check input` before reaching prompts or logs.

---

## Prompt Design

When Ollama (`gpt-oss:120b-cloud`) is available, `src/churn_assistant/llm_provider.py` prompts it with:
- A strict system instruction separating cited **signals** (`drivers`) from policy-bound **recommendations** (`recommended_action`).
- Structured JSON input containing only `verified_records`, `active_policies`, `expired_policies`, `data_conflicts`, and the already-computed `fixed_risk`.
- Output constrained to the `InsightResponse` Pydantic schema via `with_structured_output`.
- When Ollama is offline (`USE_FALLBACK_LLM=true` or server unreachable), a deterministic policy-grounded generator produces the exact schema so all 8 golden tests and the UI run reliably offline.

---

## Quick Start

### 1. Install Dependencies (`uv` or `pip`)

```powershell
# Using uv:
uv sync

# Or using venv + pip:
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Optional: Start Ollama with `gpt-oss:120b-cloud`

```bash
ollama pull gpt-oss:120b-cloud
```
*(If Ollama is not running, the assistant automatically uses its deterministic grounded fallback.)*

### 3. Run the FastAPI Service (`POST /insight`)

```powershell
uvicorn api:app --reload --port 8000
```

### 4. Run the Streamlit Agent UI

```powershell
streamlit run app.py
```

### 5. Run Golden Tests (T1–T8 + Guard & API Tests)

```powershell
pytest -v
```
