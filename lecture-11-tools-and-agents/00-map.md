# Lecture 11 Map

A one-page map for Lecture 11: **what an LLM system is now**, **where tool use sits in post-training**, **what a multi-turn agentic rollout does to lecture 3's training loop**, and **which tokens are trained on**. It's the lecture 11 companion to `../lecture-3-policy-gradients/00-training-loop-timeline.md`, `../lecture-4-implementing-rl/00-code-map.md` and `../lecture-5-reasoning-models/00-map.md`.

Covers: `01`–`04` (the whole lecture; the one-minute "Takeaways" chapter is merged into `04`).

---

## An LLM is now a system

```
                ┌──────────────────────────── HARNESS ─────────────────────────────┐
                │ system prompt + tool schemas · orchestration loop · context       │
                │ management (truncate/compact) · permissions/sandbox · subagents  │
                │                                                                  │
   user task ──►│   MODEL WEIGHTS ──tool-call tokens──►  parse ──► TOOLS           │
                │        ▲                                          (shell, files, │
                │        └──────── tool-output tokens appended ◄──── search, APIs) │
                └──────────────────────────────────────────────────────────────────┘
                benchmark score = model × harness          (MCP standardizes the TOOLS side)
```

---

## Where tool use sits in post-training

```
PRETRAINING / MID-TRAINING
     │
     ▼
SFT ON TOOL TRAJECTORIES ─── teaches format + tool selection (e.g. OpenThoughts-Agent, ~100K trajectories)
     │                       tool-output tokens MASKED from the loss
     ▼
(PREFERENCE TUNING) ─── "still works, less popular today"
     │
     ▼
RL WITH ENVIRONMENT FEEDBACK ─── prompt + ENVIRONMENT + tests (e.g. TMax: ~14.6K containers)
     │                           GRPO variants (DPPO), sometimes PPO; async actors + sandboxes
     │                           often one RL run PER DOMAIN, because of the "mixing wall"
     ▼
MULTI-TEACHER ON-POLICY DISTILLATION (MOPD) ─── merge domain experts into one model
     │
     ▼
DEPLOY inside a harness (train under several harnesses so this transfer is smooth)
```

---

## The agentic rollout on lecture 3's training loop

The step numbers match lecture 3's timeline and lecture 4's code map.

```
step   lectures 3–5 (single answer)              lecture 11 (agentic, multi-turn)
────   ──────────────────────────────            ─────────────────────────────────────────────────────────────
1      sample prompts                            sample TASKS (repo/container + instructions + hidden tests);
                                                 start G sandboxes per task (GRPO)
2      generate one completion;                  ORCHESTRATOR LOOP:
       save log-probs (π_old)                      generate until a tool call → PAUSE (GPU serves others)
                                                   → sandbox EXECUTES (CPU, seconds–minutes)
                                                   → APPEND tool output (mask 0) → RESUME (prefill) → …
                                                   → final answer (no tool call)
                                                 save token IDs + per-token engine log-probs; build STRIPED mask
3      verifier / reward model scores the text   VERIFIER RUNS IN THE ENVIRONMENT: hidden tests on the final
                                                 sandbox state → R            ("the verifier becomes an environment")
4      ref log-probs                             same (over the whole sequence; only mask-1 tokens matter)
5      advantages                                same; one advantage per rollout copied to every mask-1 token
                                                 in every turn (GRPO), or a critic (PPO, e.g. GLM-5)
6      loss × completion_mask                    same code; the mask is now 0000 1111 000 1111 00 111
7      optimizer step                            same; collapse risk past a few hundred steps (TMax)
```

**What's new is all in steps 1–3.** Steps 4–7 are lecture 4's code, with a striped mask. Details: `03` (one rollout's timing) and `04` (many rollouts, async, long tail).

---

## Which tokens are trained on

| Tokens | Written by | Read by the model? | `completion_mask` |
|---|---|---|---|
| System prompt + tool schemas | dataset / harness | yes | **0** |
| User task | dataset | yes | **0** |
| Turn headers (`<|im_start|>assistant`) | chat template | yes | **0** |
| Thinking, tool-call JSON, final answer, stop tokens | **the policy** | yes | **1** |
| Tool output (+ its wrapper tags) | **the environment** | yes | **0** |

Why tool output is masked: the model didn't choose it; training on it teaches the model to *hallucinate* tool results; and it has no π_old log-prob for a ratio. Full example in `03`.

---

## How lecture 11 connects to lectures 3–5

| Lecture 11 idea | Builds on | Where |
|---|---|---|
| Masking tool outputs | Prompt masking: "the model didn't choose the prompt" | L4 `01`, L4 `00-code-map.md` (step 2, mask bug) → `03` |
| Striped mask + per-token loss over many turns | Loss aggregation: divide by `mask.sum()` | L4 `06` → `03` |
| One advantage for a whole multi-turn rollout | "Advantage copied to every token" | L3 `03`, `05` → `03`, `04` |
| Credit assignment over 10⁵–10⁶ tokens; GLM-5 uses PPO | Critic vs. group baseline; "don't sleep on PPO" | L3 `04`, L5 `02` → `04` |
| Generate → pause → execute → resume | Step 2 of the loop | L3 `00-training-loop-timeline.md` → `03` |
| Verifier runs in the sandbox | "The verifier is becoming an environment again" | L5 `01` → `03`, `04` |
| Classical RL returns (agent acting in a world) | "Classical RL vs. RLHF" | L3 `01` → `04` |
| Actors + sandboxes, long-tail rollouts, weight sync mid-rollout | Async actors/learners, staleness, π_old | L4 `04`, L3 `04` → `04` |
| Train/inference mismatch (FP32 LM head, DPPO) | Silent-bug map; MiniMax-M1 | L4 `00-code-map.md`, L5 `02` → `04` |
| Environments must be learnable but hard | 20–80% difficulty filtering | L5 `03` #1 → `04` |
| MOPD to merge domain experts | MiMo-V2-Flash's MOPD | L5 `02` → `04` |
| Reward hacking via tools; KL coming back | Over-optimization and the KL leash | L3 `02`, L5 `03` #4, `../lecture-9-over-optimization/`, `../lecture-10-regularization/` → `04` |
| pass^k (all k succeed) | pass@k (any of k succeeds) | L5 `01` → `02` |

---

## Glossary additions (new in lecture 11)

| Term | Meaning |
|---|---|
| **Tool use** | The model emits a structured request (tool name + arguments); an orchestrator runs it; the result is appended to the context; the model continues |
| **Function calling** | Tool use where arguments must match a declared schema (usually JSON Schema) |
| **Code execution** | Tool use where the tool is a code interpreter: "the most general tool" |
| **Tool call / tool output** | The model-sampled tokens requesting an action / the environment-written tokens with the result |
| **Orchestrator** | The software loop that parses tool calls, executes them and appends results |
| **Harness** (agent scaffold) | Everything around the weights: system prompt, tool definitions, orchestration loop, context management, permissions/sandboxing, subagents. E.g. Claude Code, Codex CLI, OpenHands |
| **MCP** (Model Context Protocol) | Open standard (JSON-RPC 2.0) for exposing **resources, prompts and tools** via servers → clients → hosts |
| **ReAct** | Reasoning and acting interleaved in one generation: think → act → observe → … |
| **Toolformer** | A model inserts API calls into text and keeps those that lower perplexity, then trains on them (synthetic data) |
| **Perplexity** | How surprised a model is by text; lower = better predicted |
| **Chat template** | The model-specific special tokens that turn messages (system/user/assistant/tool) into one token sequence |
| **Masking tool outputs** | Setting `completion_mask = 0` on tool-output tokens so they're read but not trained on |
| **Striped mask** *(my term)* | The 0/1 pattern of a multi-turn rollout: alternating model and environment segments |
| **Reasoning continuity** | Whether earlier `<think>` tokens are kept in context: usually kept within a turn, often dropped across user turns |
| **Constrained decoding / strict mode** | Only allowing next tokens that keep the output valid for a schema |
| **Compaction** | The harness summarizing/truncating history to free context space |
| **Environment** (agentic RL) | A sandbox (container/microVM) with files, services, tools and verification tests |
| **Sandbox** | The isolated CPU machine an agent acts in during a rollout |
| **Outcome-only reward** | One reward at the end of a trajectory (e.g. tests pass) |
| **pass^k** | Chance that **all** k tries succeed (reliability; τ-bench). Contrast pass@k |
| **Terminal-Bench 2.0** | End-to-end terminal-agent tasks in containers, checked by tests |
| **Long-tail rollouts** | Very uneven rollout lengths (2 vs. 200 tool calls) that stall synchronous batches |
| **Train/inference mismatch** | The inference engine and trainer computing different log-probs for the same tokens |
| **DPPO** | A GRPO variant for agentic stability that masks tokens where trainer and engine diverge (see `../lecture-10-regularization/`) |
| **Harness randomization** | Training under several harnesses so the model doesn't overfit to one |
| **RLVR mixing wall** | With many domains in one RL run, each gets too few samples per batch; fix: per-domain RL + MOPD |
