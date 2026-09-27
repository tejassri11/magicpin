# VERA Bot — magicpin AI Challenge

FastAPI backend implementing the 5-endpoint judge contract for VERA, magicpin's merchant AI assistant.

---

## Folder structure

```
vera-bot/
├── main.py               # FastAPI app + all 5 endpoint handlers
├── state.py              # Thread-safe in-memory context/suppression/opt-out store
├── conversation.py       # Per-conversation turn history and lifecycle state
├── decision.py           # SEND vs SKIP engine (10 ordered gate checks, no LLM)
├── composer.py           # Context assembler + LLM router + stub fallback
├── reply_handler.py      # Multi-turn reply: auto-reply detection, intent, composition
├── models.py             # Pydantic v2 models for all 5 endpoint schemas
├── config.py             # All settings from environment variables
├── logger.py             # Structured stdout logger
│
├── prompts/
│   ├── system_prompt.py  # Vera persona, grounding rules, anti-patterns
│   ├── trigger_prompts.py# 15+ trigger-kind-specific prompt variants
│   └── reply_prompts.py  # Reply variants (action mode, curveball, graceful exit)
│
├── tests/
│   ├── test_state.py     # Unit tests for state store
│   └── test_endpoints.py # Integration tests for all 5 endpoints
│
├── .env.example          # Template for secrets / settings
├── requirements.txt
└── README.md
```

---

## Quick start

### 1. Install dependencies

```bash
cd vera-bot
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and set:
#   LLM_PROVIDER=anthropic
#   LLM_API_KEY=sk-ant-...
#   TEAM_NAME=Your Team
#   CONTACT_EMAIL=you@example.com
```

Without `LLM_API_KEY`, the bot runs in **stub mode** — every endpoint works, but
the composed messages are templated (not LLM-generated). Useful for verifying the API
contract before adding the LLM key.

### 3. Run locally

```bash
cd vera-bot
python main.py
```

Or with uvicorn directly:

```bash
uvicorn main:app --host 0.0.0.0 --port 8080 --reload
```

Server starts at `http://localhost:8080`.

### 4. Run tests

```bash
cd vera-bot
python -m pytest tests/ -v
```

---

## Endpoint reference

### GET /v1/healthz

```bash
curl http://localhost:8080/v1/healthz
```

Response:
```json
{
  "status": "ok",
  "uptime_seconds": 42,
  "contexts_loaded": {
    "category": 5,
    "merchant": 50,
    "customer": 200,
    "trigger": 100
  }
}
```

Always HTTP 200. Never uses LLM. Responds in < 100ms.

---

### GET /v1/metadata

```bash
curl http://localhost:8080/v1/metadata
```

Response:
```json
{
  "team_name": "Team Vera",
  "team_members": ["Your Name"],
  "model": "claude-sonnet-4-5",
  "approach": "Trigger-kind-routed composer with grounded prompts...",
  "contact_email": "team@example.com",
  "version": "1.0.0",
  "submitted_at": "2026-09-27T00:00:00Z"
}
```

---

### POST /v1/context

Push a category, merchant, customer, or trigger context. Idempotent by (context_id, version).

```bash
# Push a category
curl -X POST http://localhost:8080/v1/context \
  -H "Content-Type: application/json" \
  -d '{
    "scope": "category",
    "context_id": "dentists",
    "version": 1,
    "payload": { "slug": "dentists", "voice": { "tone": "peer_clinical" } },
    "delivered_at": "2026-09-27T00:00:00Z"
  }'

# Response (accepted):
# {"accepted": true, "ack_id": "ack_dentists_v1", "stored_at": "..."}

# Push same version again (idempotent):
# {"accepted": false, "reason": "stale_version", "current_version": 1}

# Push older version:
# {"accepted": false, "reason": "stale_version", "current_version": 1}

# Push newer version (replaces):
# {"accepted": true, "ack_id": "ack_dentists_v2", "stored_at": "..."}
```

**Valid scopes**: `category`, `merchant`, `customer`, `trigger`

---

### POST /v1/tick

Wake-up call. Bot evaluates available triggers and returns proactive messages.

```bash
curl -X POST http://localhost:8080/v1/tick \
  -H "Content-Type: application/json" \
  -d '{
    "now": "2026-09-27T00:00:00Z",
    "available_triggers": ["trg_001_research_digest_dentists"]
  }'
```

Response:
```json
{
  "actions": [
    {
      "conversation_id": "conv_m_001_trg_001_abc12345",
      "merchant_id": "m_001_drmeera_dentist_delhi",
      "customer_id": null,
      "send_as": "vera",
      "trigger_id": "trg_001_research_digest_dentists",
      "template_name": "vera_research-digest_v1",
      "template_params": ["Dr. Meera", "high-risk adult patients"],
      "body": "Dr. Meera, a new JIDA paper...",
      "cta": "open_ended",
      "suppression_key": "research:dentists:2026-W17",
      "rationale": "Research digest trigger for dentist category..."
    }
  ]
}
```

Returns `{"actions": []}` when no triggers pass the SEND/SKIP gates.

---

### POST /v1/reply

Receive a merchant/customer reply and get the bot's next action.

```bash
curl -X POST http://localhost:8080/v1/reply \
  -H "Content-Type: application/json" \
  -d '{
    "conversation_id": "conv_m_001_trg_001_abc12345",
    "merchant_id": "m_001_drmeera_dentist_delhi",
    "from_role": "merchant",
    "message": "Yes please go ahead!",
    "received_at": "2026-09-27T00:05:00Z",
    "turn_number": 1
  }'

# Response (send):
# {"action": "send", "body": "...", "cta": "binary_yes_no", "rationale": "..."}

# Response (wait):
# {"action": "wait", "wait_seconds": 1800, "rationale": "..."}

# Response (end):
# {"action": "end", "rationale": "..."}
```

**Intent detection**:
- `yes / haan / chaliye / ok go ahead` → explicit commit → action mode
- `not interested / stop / band karo` → hard reject → end + 30-day opt-out
- `Thank you for contacting us...` → auto-reply → probe → wait → end
- `later / busy / baad mein` → wait 30 minutes

---

### POST /v1/teardown (optional)

Wipe all state at end of test run:

```bash
curl -X POST http://localhost:8080/v1/teardown
```

---

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `anthropic` | `anthropic` \| `openai` \| `gemini` \| `deepseek` |
| `LLM_MODEL` | `claude-sonnet-4-5` | Model name for the chosen provider |
| `LLM_API_KEY` | *(empty)* | API key — leave empty for stub mode |
| `LLM_TIMEOUT_SECONDS` | `25` | LLM call timeout (5s below judge's 30s limit) |
| `LLM_MAX_TOKENS` | `600` | Max tokens per LLM call |
| `TEAM_NAME` | `Team Vera` | Returned by `/v1/metadata` |
| `TEAM_MEMBERS` | `Builder` | Comma-separated, returned by `/v1/metadata` |
| `CONTACT_EMAIL` | `team@example.com` | Returned by `/v1/metadata` |
| `MAX_ACTIONS_PER_TICK` | `20` | Cap on actions per `/v1/tick` (challenge max = 20) |
| `OPTOUT_COOLDOWN_DAYS` | `30` | Days to suppress after merchant opts-out |
| `SOFT_TURN_LIMIT` | `5` | Gracefully close conversation after N turns |
| `HARD_TURN_LIMIT` | `7` | Force-close conversation after N turns |
| `PORT` | `8080` | Server port |
| `LOG_LEVEL` | `INFO` | `DEBUG` \| `INFO` \| `WARNING` \| `ERROR` |

---

## Running against the judge simulator

```bash
# 1. Start the bot
cd vera-bot
python main.py

# 2. In another terminal, run the simulator
cd ..                         # back to challenge root
python judge_simulator.py
```

The simulator reads `BOT_URL` from its config section. Set it to `http://localhost:8080`.

### Iteration loop

```
1. Run "warmup" scenario    → fix endpoint errors
2. Run "auto_reply_hell"    → tune auto-reply detection
3. Run "intent_transition"  → tune commit phrase list
4. Run "hostile"            → verify graceful exit
5. Run "phase2_short"       → verify LLM scoring
6. Run "full_evaluation"    → baseline score; tune prompts
```

---

## Architecture overview

```
Judge → POST /v1/context  →  StateStore.put_context() [versioned, atomic]
Judge → POST /v1/tick     →  DecisionEngine × N triggers (parallel)
                                    → ComposerPipeline → LLM (temp=0)
                                    → ConversationStore.start()
                                    → SuppressionStore.fire()
Judge → POST /v1/reply    →  AutoReplyDetector → IntentClassifier
                                    → ReplyComposer → LLM
                                    → ConversationStore.update()
```

All state is in-memory. No external database. Safe for the duration of a single judge test run.
#   m a g i c p i n  
 