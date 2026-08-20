# Local UI — a browser workstation for the real agent

A localhost-only web interface to the agent that already lives in this repo. It is a **client** of
the existing stack, not a second implementation: chat drives the real `run_session` loop, tool calls
go through the real controller (`execute_proposal`), external text is sanitised by the real trust
boundary, results are hashed into the real telemetry ledger, and every claim is graded by the real
verifier. The browser is a window onto that machinery — the backend stays authoritative.

There is **no fake chatbot and no mock tools** anywhere in the running app. Mocks exist only in the
test suite, and the one non-model shortcut — a `--stub` echo generator for exercising the UI without a
GPU — announces itself as a stub in the status, the banner, and every reply it produces.

---

## What you get

- **Chat** with the agent — it reasons, then *proposes* tools.
- **Human-in-the-loop approval.** Every tool call surfaces as an Approve/Deny card. Nothing runs
  until you approve it, and approval is enforced on the backend (the client cannot bypass it).
- **Live activity feed.** THINKING → PLAN → PROPOSAL → AUTHORIZATION → TOOL RESULT (sha256) →
  INTERPRET → VERIFICATION → VERDICT, streamed as the turn happens. This is the telemetry ledger,
  rendered.
- **Findings** graded as hypotheses until a validating test confirms them, with severity, status,
  affected component, and the next test that would raise confidence.
- **Session control** — start an operator-authorized session (objective, targets, capability
  profile), then converse without re-approving the session each turn.
- **Kill switch** — one button engages the file-based kill switch; it halts turns on the backend,
  and clearing it requires an explicit operator acknowledgement.

---

## Requirements

Everything is already pinned by the repo. The UI adds only FastAPI + a server:

| | |
|---|---|
| Python | 3.11+ (developed on 3.13.7) |
| Web | `fastapi`, `uvicorn`, `starlette`, `websockets` (already present in `.venv`) |
| Model (real) | a **CUDA GPU**, plus the repo's training/serving stack (`transformers==4.57.6`, `bitsandbytes`, `torch`, `peft`) |
| Frontend | none to build — a single self-contained `webui/static/index.html` (no bundler, no CDN, no external assets) |

There is nothing to `npm install`. The frontend is one hand-written file served by the backend.

---

## Starting it

From the repo root:

```bash
python start_local.py            # loads the real model if a GPU is present, else says why
python start_local.py --stub     # clearly-labelled echo model — exercise the UI without a GPU
python start_local.py --port 8010 --model qwen
```

Windows convenience wrappers (they prefer `.venv\Scripts\python.exe`):

```bash
start_local.bat
```

```bash
start_local.ps1 -- --stub --port 8010
```

The launcher prints an honest banner (backend/frontend/model/device/RAG/memory/tools/verifier/trust
boundary) and the local URL, then serves on **127.0.0.1 only**. Open the URL it prints
(default `http://127.0.0.1:8000`).

### `--stub` vs the real model

- **On a GPU host, without `--stub`:** the model loads through the existing lock seam
  (`serving.model_spec.load_lock` + the same 4-bit `BitsAndBytesConfig` the benchmarks use, greedy
  decode). Status reads `ready` and names the device.
- **On a CPU laptop:** `torch.cuda.is_available()` is `False`, so the real load is refused honestly —
  status reads `gpu_required` and the chat endpoint returns an error with that reason rather than
  faking a reply. Use `--stub` to click through the whole UI, or run on a GPU host.

The stub is a deliberate echo: if your message hints at reviewing source it proposes a read-only
`source_scan` so the approval/execution/verification path is exercised; otherwise it echoes your
message prefixed with `[STUB MODEL — not a real model]`. A yellow banner says the same. It can never
be mistaken for the real model.

---

## Using it

1. **(Optional) Start a session.** Pick a capability profile (`read_only`, `recon`, `validation`,
   `full`), optionally list authorized targets and an objective, and click *Start session*. The
   operator at localhost is the human gate — starting a session authorizes the declared targets once,
   so you are not re-prompted for the session on every turn.
2. **Chat.** Type a request and send. The agent reasons and may propose a tool.
3. **Approve or deny.** Each proposal is a card. Approve to run the real tool; deny to skip it. The
   turn continues either way and ends with a verified verdict.
4. **Read the results.** The Activity tab is the live event stream; the Findings tab lists graded
   hypotheses. Reports are written to `data/webui/reports/` and listed via the API.
5. **Stop anytime.** *STOP (kill switch)* halts the backend immediately; *Resume* clears it (operator
   acknowledgement).

---

## Architecture — how the UI connects to the agent

```
browser (index.html, one file)
   │  REST: /api/status /api/tools /api/profiles /api/session/* /api/findings /api/reports ...
   │  WS  : /ws   {action: chat|approve|deny|stop}
   ▼
webui/server.py         FastAPI adapter. 127.0.0.1 only. Reuses existing modules; never reimplements.
   │                    A chat runs a Turn in loop.run_in_executor; events stream back over the WS via
   │                    loop.call_soon_threadsafe. approve/deny resolves the pending proposal.
   ▼
webui/runner.py (Turn)  Builds a Telemetry("ui") whose sink emits each record to the browser, then
   │                    calls the REAL research.session.run_session(question, generate, approver, ...).
   │                    The approver emits an "approval_required" event and BLOCKS on a threading.Event
   │                    until the browser sends approve/deny — that is the human gate.
   ▼
research.session.run_session      the real agent loop (plan → propose → authorize → execute → verify)
   ├─ webui/model.py get_generate()  the injectable generate(messages)->str seam:
   │                                  real model on GPU, honest "gpu_required" on CPU, --stub echo.
   ├─ mcp_layer execute_proposal     real tool execution — still checks operator_ack + kill switch,
   │                                  tools stay gated by capability profile.
   ├─ trust/boundary.py              sanitises all external/tool content before the model sees it.
   ├─ mcp_layer/telemetry.py         hashes tool_result (sha256) into the ledger = the activity feed.
   └─ verification/verify.py         grades the final answer (overclaim / severity / authorization).
```

The single reuse point is the `generate(messages) -> str` seam that `run_session` and
`run_assessment` already accept. The UI supplies that callable (real, stub, or — in tests — a mock).
Nothing else about the agent changes.

**Files added**

```
webui/server.py            FastAPI app: REST + /ws WebSocket, findings collection, report save
webui/runner.py            Turn: telemetry→browser sink, blocking human-approval callback
webui/model.py             the generate() seam (real GPU / honest CPU / labelled stub)
webui/static/index.html    self-contained dark "workstation" SPA (no build, no external assets)
start_local.py             launcher (--host/--port/--stub/--model); prints honest banner
start_local.bat            Windows wrapper
start_local.ps1            PowerShell wrapper
tests/api/test_webui.py    FastAPI TestClient API tests + the E2E (chat→propose→approve→run→verify)
```

---

## Security notes

- **Localhost only.** The server binds `127.0.0.1`. There is no cloud, no public URL, no external
  service — except a tool you explicitly authorize and approve.
- **The backend is authoritative.** Approval is enforced server-side; a crafted client message cannot
  make a tool run without an approval reaching the blocking callback. `execute_proposal` still checks
  `operator_ack` and the kill switch on every call.
- **Tools stay gated** by the session's capability profile (`read_only` / `recon` / `validation` /
  `full`). The profile is chosen by the operator, not the model.
- **External content is untrusted.** Everything a tool returns is routed through the trust boundary
  before it reaches the model, so page/file/response text cannot smuggle instructions into the loop.
- **No secrets on the wire.** The telemetry sink redacts secrets, and tool results are shown by
  sha256 digest in the feed. Status never exposes environment variables or API keys.
- **Rendering is XSS-safe.** The SPA escapes HTML first, then applies a tiny, fixed Markdown subset —
  model output and tool text cannot inject markup.
- **Kill switch is real.** It is the file-based switch the rest of the system honours; engaging it
  from the UI halts turns on the backend, and clearing it takes an operator acknowledgement.

---

## Troubleshooting

| symptom | cause / fix |
|---|---|
| status shows `gpu_required`, chat returns an error | No CUDA GPU on this host. Use `--stub`, or run on a GPU host. This is expected on a laptop. |
| status shows `error: transformers 5.x …` | 5.x cannot quantise this model. `pip install transformers==4.57.6` (see the main README). |
| port already in use | `--port 8010` (or any free port). |
| `RAG: not_built` in the banner | `rag/index` has not been built yet; chat still works, retrieval is just empty. |
| the stub keeps re-proposing the same tool | Expected — the stub is a naive echo; the real model converges. Deny to end the turn. |
| nothing streams in the Activity panel | The WebSocket did not connect; confirm the URL/port and that the server is still running. |

---

## Verification (what was actually tested)

- **`tests/api/test_webui.py`** (in the regression gate): status is honest (34 tools, CUDA `False` on
  this host); chat with no model errors honestly; session start is operator-gated and profile-enforced;
  the **E2E** sends a chat, receives a real tool proposal, approves it from the "browser", and confirms
  the real `source_scan` ran through the real controller + trust boundary + verifier (telemetry
  `tool_result` + `verification` present, turn `completed`); the deny path records the proposal as
  declined; the kill switch blocks a turn; malformed model output still completes.
- **Live browser run** (in-app browser → `http://127.0.0.1`): the served SPA drove a real turn over the
  WebSocket — THINKING → PLAN → **APPROVED** → AUTHORIZATION(approved) → **TOOL RESULT (sha256)** →
  **DENIED** → AUTHORIZATION(declined) → INTERPRET → **VERIFICATION PASS** → VERDICT, ending with the
  assistant's answer. Both the approve and deny paths were exercised through the actual UI, not a
  TestClient.

The full gate is `python scripts/run_tests.py` (46 modules; all pass).

---

## Honest limitation

The **real model is verified only through the seam on this laptop**, because
`torch.cuda.is_available()` is `False` here — so the end-to-end runs above used the labelled stub. The
identical seam loads the real model on a GPU host (the same lock + 4-bit config the benchmarks use);
that path is exercised by `webui/model.load()` and gated to `gpu_required` on CPU. Run
`python start_local.py` on a GPU host for a real-model conversation.
