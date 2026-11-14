# Part 2: Infra: How Tool Calls Actually Work

Notes on *How Language Models Use Tools and the Path to Agents | Post-Training Course, Lecture 11* (Nathan Lambert), 15:30–24:17. Based on `source/lecture-11-tools-and-agents-slides.md` and `source/lecture-11-tools-and-agents-transcript.md`. Follows on from `02-why-language-models-need-tools.md`. See `00-map.md` for the rollout timeline on the training loop.

---

## Where this fits

This is the section to know if you're going to RL-train a coding agent. It answers three questions:

1. **What does a tool call look like at the token level?** (One stream, two authors.)
2. **Who runs the tool?** (Not the model. An orchestrator / harness.)
3. **Which tokens does training learn from?** (Only the ones the model sampled. Tool outputs are **masked out of the loss**.)

Lambert is upfront that he's "not an expert" on the harness side, and that the time delay while a tool runs "makes for a very complex inference situation that we don't talk about in this lecture." These notes fill in some of that with labelled additions, because it's the part that matters most for training.

---

## Slide: "One token stream: tools add tokens mid-generation"

*(The figure isn't in the sources. Reconstructed from the slide caption and transcript.)*

```
 prompt tokens   model tokens   TOOL CALL (orange)   [pause]   TOOL OUTPUT (purple)   model tokens   TOOL CALL ...   final answer
 ─────────────   ────────────   ──────────────────             ────────────────────   ────────────   ─────────────   ────────────
  given           sampled        sampled by model,              written by an          sampled         sampled          sampled
                                 flagged for execution          external system
```

The caption: "The model generates until it emits a tool call (orange); an external system executes it and injects the output (purple) into the sequence; the model continues. Multiple tool calls can occur in a single generation. **During training, tool call outputs are masked from the loss.**"

Lambert's walk-through:
- The orange tokens "are still generations from a model." When they're inside the code bracket, "it gets flagged to pass to an external evaluator."
- "There is a **time delay** here." The tool takes real wall-clock time to run.
- After the output arrives, "the loop is handed back to the language model, which starts generating tokens again."
- It can cycle: "language model, tool, language model, tool, language model, tool, final answer."

---

## Slide: "The model only sees tokens: tools live in the system prompt"

Tool-use training data looks like ordinary post-training data with **one addition: a system prompt that declares the available tools as JSON schemas.**

```xml
<system>
You are a function-calling AI model. You are
provided function signatures within <functions>
XML tags. You may call one or more functions.
</system>

<functions>
[{
  "name": "search_movies",
  "description": "Search movies by title.",
  "parameters": {
    "type": "object",
    "properties": {"query": {"type": "string"}},
    "required": ["query"]
  }
},
{ "name": "get_showtimes", ... },
{ "name": "get_movie_details", ... }]
</functions>
<user> ... </user>
```

- **"The model never 'connects' to anything."** It learns to emit text matching the declared schemas, and to expect results to appear in its context.
- **Open models must generalize to arbitrary tools** that users declare. Lambert returns to this often: open models get used "off the shelf" in someone else's environment. ChatGPT and Claude Code are "a vertical experience" where the harness and model are built together, which is an advantage closed labs have.

*(My note: the tool schemas are part of the **prompt**, so in RL they're masked out like any other prompt token. They can also be long: dozens of tools can mean thousands of prompt tokens on every rollout.)*

---

## Slide: "An orchestrator sits between the model and its tools"

```python
messages = [...]
while True:
    resp = model(messages, tools=tools)
    if not resp.tool_calls:
        return resp.text

    messages.append(resp.message)
    for call in resp.tool_calls:
        result = execute_tool(
            call.name, call.args)
        messages.append({
            "role": "tool",
            "tool_call_id": call.id,
            "content": result})
```

"The most basic harness... is a while loop that tells the model when to generate and tells the tools when to execute and stacks them all together. That is the simplest harness in the world."

- **The model's only power is emitting tokens.** The orchestrator **parses** them, **executes** the tools and **appends** results.
- **Training for tool use = making the model behave predictably under this altered token flow:** when to call, how to format arguments, how to use the results.

### The same loop at the token level *(my addition)*

The slide's loop works with **messages** (an API view). During RL training you need the **token** view, because the loss is per token. Here's one complete rollout for a coding agent, in a simplified Qwen-style chat template. (Illustrative: exact special tokens, role names and newlines differ by model. Qwen 2.5/3, for example, wraps tool output in `<tool_response>` tags inside a `user`-role turn; other templates use a `tool` role. Always check your model's chat template.)

```
seg  text                                                                       written by             mask
───  ─────────────────────────────────────────────────────────────────────────  ─────────────────────  ────
S    <|im_start|>system                                                         dataset + harness      0
     You are a coding agent. # Tools <tools>{"name":"bash",                     (tool schemas)
       "parameters":{"cmd":{"type":"string"}}}</tools><|im_end|>
U    <|im_start|>user                                                           dataset (the task)     0
     tests/test_dates.py fails. Fix it.<|im_end|>
H1   <|im_start|>assistant                                                      chat template          0
A1   <think>Run the failing test first.</think>                                 ★ MODEL (sampled) ★    1
     <tool_call>{"name":"bash","arguments":{"cmd":"pytest tests/test_dates.py
       -q"}}</tool_call><|im_end|>                       ← stop token: also sampled
                                ── generation pauses; harness parses A1, runs pytest in the sandbox ──
T1   <|im_start|>user <tool_response>                                           ENVIRONMENT (pytest)   0
     FAILED test_parse - assert '2024-1-5' == '2024-01-05'                      + template wrapper
     </tool_response><|im_end|>
H2   <|im_start|>assistant                                                      chat template          0
A2   <think>Missing zero-padding. Fix the format string.</think>                ★ MODEL (sampled) ★    1
     <tool_call>{"name":"bash","arguments":{"cmd":"sed -i 's/%-m/%m/;
       s/%-d/%d/' src/dates.py && pytest -q"}}</tool_call><|im_end|>
                                ── pause; harness runs the edit + tests ──
T2   <|im_start|>user <tool_response>3 passed in 0.12s</tool_response><|im_end|>  ENVIRONMENT            0
H3   <|im_start|>assistant                                                      chat template          0
A3   Fixed: dates are now zero-padded in src/dates.py. All 3 tests pass.        ★ MODEL (sampled) ★    1
     <|im_end|>                                          ← no tool call → rollout ends
                                ── verifier runs hidden tests in the final sandbox state → reward R = 1 ──
```

**The rule for the mask:** `completion_mask = 1` **exactly on tokens the policy sampled** (A1, A2, A3, including each `<|im_end|>` it chose to emit). Everything else is 0: the prompt (S, U), the turn headers the template inserts (H1–H3), and **all tool output plus its wrapper** (T1, T2).

| Segment | Who wrote it | In the forward pass? (model attends to it) | In the loss? |
|---|---|---|---|
| System + tool schemas, user task | Dataset / harness | ✅ yes | ❌ mask 0 (same as lecture 4's prompt masking) |
| Turn headers (`<|im_start|>assistant`) | Chat template | ✅ yes | ❌ mask 0 |
| **Thinking, tool calls, final answer, stop tokens** | **The policy** | ✅ yes | ✅ **mask 1** |
| **Tool output** (pytest log, file contents, search results) | **The environment** | ✅ yes | ❌ **mask 0** |

The second column is the part that's easy to miss: **masking doesn't hide the tool output from the model.** The model still reads it (it's in the context, so A2's tokens are predicted *given* T1). Masking only means "don't push up the probability of these tokens." That's exactly how the prompt already works in lecture 4.

Connects to: `../lecture-4-implementing-rl/00-code-map.md` (step 2 produces `completion_mask`; the silent-bug row "prompt or padding tokens left in `completion_mask`") and `../lecture-4-implementing-rl/01-translating-the-policy-gradient-into-code.md` ("the model didn't choose the prompt, it was given it"). A multi-turn rollout's mask isn't one block of 0s followed by one block of 1s. It's **striped**: `0000 1111 000 1111 00 111`.

#### Building the striped mask during the rollout *(my addition, a sketch)*

```python
ids, mask, old_logps = [], [], []

def append(tokens, sampled, logps=None):
    ids.extend(tokens)
    mask.extend([1 if sampled else 0] * len(tokens))
    old_logps.extend(logps if sampled else [0.0] * len(tokens))   # env tokens: placeholder, masked anyway

append(tokenize(system + user) + header("assistant"), sampled=False)    # S, U, H1
for turn in range(MAX_TURNS):
    out = engine.generate(ids, stop=["<|im_end|>"])        # A_k: token ids + their log-probs (π_old)
    append(out.token_ids, sampled=True, logps=out.logprobs)
    call = parse_tool_call(out.text)
    if call is None:
        break                                               # final answer: rollout done
    result = sandbox.run(call)                              # seconds to minutes, on a CPU
    append(tokenize(wrap_tool_output(result)) + header("assistant"), sampled=False)   # T_k, H_{k+1}

reward = verifier(sandbox)                                  # e.g. hidden unit tests
```

Two practical points that follow from this:
- **Keep the token IDs exactly as sampled.** If you instead save the messages as text and re-render them through the chat template later, the tokenizer can split the text differently, or the template can drop things (see "reasoning continuity" below). Then the sequence you train on isn't the sequence you sampled, and the saved `old_logps` no longer line up, so ρ is wrong. This is the same family of bug as the code map's "inference engine vs. trainer log-prob mismatch."
- **Only the sampled tokens have meaningful `old_logps`.** The environment's tokens were never sampled by π_old, so there's no ratio ρ to compute for them. That's a third reason (beyond the two below) they must be masked.

### Worked example: what the mask does to the numbers *(my addition)*

Say the rollout above has these token counts:

| Segment | Tokens | Mask |
|---|---|---|
| S + U (incl. tool schemas) | 350 | 0 |
| H1 + H2 + H3 | 9 | 0 |
| A1 + A2 + A3 (model) | 40 + 45 + 20 = **105** | 1 |
| T1 + T2 (tool output) | 55 + 20 = 75 | 0 |
| **Total sequence** | **539** | |

- The forward pass (step 6) runs over all **539** positions, because the model needs the whole context.
- Only **105** per-token losses survive `× mask`. With per-sequence normalization the loss divides by `mask.sum() = 105`, not 539 (lecture 4 `06`).
- With GRPO, R = 1 becomes one advantage for the whole rollout, **copied to all 105 model tokens across all three turns**, including the thinking and tool calls in turns 1 and 2. That's lecture 3's "advantage copied to every token," now across turns.
- Real coding rollouts are much more lopsided. A single `cat` of a big file can be 10,000 tool-output tokens, so model tokens can easily be under 10% of the sequence. You pay for those tokens in memory and compute (they're in the context), but they contribute nothing to the gradient.

### One rollout's timing, mapped to lecture 3's steps *(my addition)*

In lectures 3–5, step 2 ("generate") was one uninterrupted call to the inference engine. In agentic RL, step 2 becomes **the orchestrator loop**, and step 3 ("score") becomes **running a verifier inside the environment the model just changed**.

```
lecture 3 step:  │◄──────────────────────────── 2. GENERATE (now a loop) ────────────────────────────►│◄─ 3. SCORE ─►│

wall-clock time ─────────────────────────────────────────────────────────────────────────────────────────────────►

inference GPU    [prefill S,U][sample A1]·······idle for this rollout·······[prefill T1][sample A2]··idle··[prefill T2][sample A3]
                              stop at                                                stop at                        no tool call
                              </tool_call>                                           </tool_call>                   → done
                                   │                                                     │                              │
harness          parse A1 ─────────┘                                parse A2 ────────────┘                              │
                         │   append T1 + H2 ─────────────────────┐          │   append T2 + H3 ──┐                       │
                         ▼                                       │          ▼                    │                       ▼
sandbox (CPU)    ········[run pytest: 4 s]───────────────────────┘··········[run sed + pytest: 6 s]┘··········[hidden tests] → R
                                                                                                                         │
mask built       000000 111111          0000000                 11111111     00000              111                      │
                                                                                                                         ▼
                                                                                              4. ref log-probs → 5. advantages
                                                                                              → 6. loss (masked) → 7. update
```

- **Generate → pause → execute → append → resume → … → final reward.** The reward only exists at the very end, after the last turn, so steps 4–7 can't start until the whole trajectory is done (same as lecture 3: GRPO needs all G rewards first).
- **"Resume" means a new prefill.** The engine has to process the appended tool-output tokens before it can sample again. *(Inference engines like vLLM can cache the earlier prefix so only the new tokens are processed, if the KV cache for that rollout hasn't been evicted during the pause.)*
- **While one rollout waits on its sandbox, the GPU serves other rollouts.** And while the GPU generates, the sandbox sits idle. That's the core scheduling problem of `04` (Kimi K3: sandboxes idle "up to 98% of their lifetime").
- **Steps 4–7 are unchanged** except that every per-token quantity (log-probs, ratio, KL, loss) is multiplied by the striped mask.

Connects to: `../lecture-3-policy-gradients/00-training-loop-timeline.md` (steps 2–3) and `../lecture-5-reasoning-models/01-introduction-and-core-concepts.md` ("the verifier is becoming an environment again"). Here it literally is: the reward is computed from the **state of the sandbox the model edited**, not from the text alone.

---

## Slide: "MCP: Standardizing the tool side"

**Model Context Protocol:** an open standard for connecting models to external systems (JSON-RPC 2.0 underneath).

| Server primitive | What it is |
|---|---|
| **Resources** | Read-only data blobs (e.g. a searchable document) |
| **Prompts** | Templated messages and workflows |
| **Tools** | Functions the model can call |

**Architecture:** **servers** wrap a capability → **clients** aggregate servers → **hosts** (Claude, ChatGPT apps) provide the interface. Tool developers build one interface, and swapping model vendors just means swapping the middle layer.

An MCP tool definition (from the slide) is just a schema like the system-prompt ones above:

```json
{"name": "get_weather", "description": "Get current weather",
 "inputSchema": {"type": "object",
                 "properties": {"location": {"type": "string", "description": "City or coordinates"}},
                 "required": ["location"]}}
```

Lambert is lukewarm: "the MCP popularity was kind of annoying to me... this is not that big of a deal." It's "nice to do but far from required, especially as the models keep getting smarter." **For training:** "you wanna make sure that your training data has consistent format and that your model can generalize to the different domains" your customers care about. You don't care about MCP specifically "unless your customers specifically ask for it."

---

## Slide: "What is a harness?"

MCP standardized the **tool** side. The **harness** (or agent scaffold) is everything around the weights on the **model** side:

- The **system prompt and tool definitions**
- The **orchestration loop** itself
- **Context management**: truncation, compaction, memory across long tasks (Lambert adds skills, and possibly other models that manage the context)
- **Permissions and sandboxing**: what the agent may touch (API keys, web access, user credentials)
- **Subagents** and parallel workstreams

...and it's "growing in complexity." Examples: **Claude Code, Codex CLI, OpenHands** (he also lists the Codex app, Pi and Hermes).

**The boxed point: benchmark scores are model × harness scores.** "The same models behave very differently inside different ones." Before, benchmarking meant getting the sampling settings right. Now "harnesses can genuinely improve performance in a substantial way." `04` gives a number for this (Polar: +22.6 points from training in an unfamiliar harness).

*(My note: for RL training, the harness **is** part of the environment. The system prompt, the tool set, the output wrapper format and the truncation rules all decide which tokens the model sees, so the harness you train in shapes the policy you get.)*

---

## Slide: "Implementation details are everywhere"

Five details, each of which can silently break training.

### 1. Masking tool outputs

"Tool-output tokens are masked from the training loss: **the model must not learn to *predict* the external system.**" Lambert: you mask them "so that the model doesn't learn to get the next-token prediction on something that was generated by a computer in a deterministic way." This applies to **both SFT and RL**.

Why, in plain terms *(my addition, expanding his point)*:

| Reason | What would go wrong without the mask |
|---|---|
| **The model didn't choose those tokens** | Same as the prompt (lecture 4 `01`). In RL, "−advantage × log π(token)" only makes sense for tokens the model actually picked. Pushing up "the pytest log" is meaningless |
| **It would learn to hallucinate tool results** | In SFT, training to predict outputs teaches the model to *imagine* what `pytest` would print, instead of waiting for it. That's the exact failure tools were meant to fix (`02`'s π example) |
| **There's no π_old for them** | The engine never sampled them, so there's no saved log-prob and no meaningful ratio ρ (see the sketch above) |

### 2. Reasoning continuity

Do `<think>` tokens from earlier steps stay in the context?

| Where | Common practice | Why |
|---|---|---|
| **Between tool calls within one turn** (think → tool → think → tool) | **Kept.** "I'm pretty sure they all keep that thinking" | The model needs its own plan to continue |
| **Across user turns** (a multi-turn chat) | **Usually erased** to cut serving cost. Some models (Kimi K3) keep the history | Efficiency "goes way up," but performance might be lower |

The training-relevant point: "You need to make sure you manage all this information **consistently across your training run and inference**." *(My note: if your chat template strips old `<think>` blocks when rendering a conversation, but your rollout kept them in the context while sampling, then re-rendering for training produces a different sequence than the one sampled. Build the training sequence from the token IDs you actually sampled, as above.)*

### 3. Formats fragment

Python-style vs. JSON calls; OpenAI's `tool_calls` vs. Anthropic's `tool_use` blocks vs. Gemini's function-calling modes. Chat templates hide this at the token level. Lambert: "know which format your model is doing and get it consistent through the whole process." People often pick the format with **the lowest token usage** for their tokenizer, to save money.

### 4. Schema conformance

Production systems can enforce valid JSON with **constrained decoding** ("strict mode"): at each step, only allow next tokens that keep the output valid for the schema. Example *(my addition)*: after `{"name": "`, only tokens that start a declared tool name are allowed. Closed labs also post-trained for flags like this, but Lambert hears less about it now that models "are just much more reliable, especially with RLVR that eliminates a lot of these issues."

*(My note: if you use constrained decoding during RL rollouts, the sampled tokens come from a **modified** distribution, not the raw policy. The log-probs the engine reports may then not match what the trainer computes, which is another train/inference mismatch to watch for.)*

### 5. Context consumption

Search and retrieval outputs can **flood the context window**. Truncate, summarize or paginate. Lambert mentions `/compact` (compacting history so the model "has more room to operate"): early Claude Code models were "really, really bad at compaction," and newer ones (he names Claude Fable and GPT-5.3) are better. Context management was "a huge problem" early on, and the harness decides how context is presented.

*(My note: truncation and compaction during an RL rollout change what's in the sequence, so they're part of the environment too. If the harness summarizes and replaces history mid-rollout, the "one striped sequence" picture breaks into several sequences, each with its own prompt. How you train on that is a design choice your RL framework has to make.)*

---

## The whole section in five lines

1. **One token stream, two authors.** The model samples text that includes tool calls; the harness pauses generation, runs the tool and appends its output; the model resumes. It can cycle many times before a final answer.
2. **Tools live in the system prompt as schemas.** The model never connects to anything. A while-loop orchestrator is the simplest harness; MCP standardizes the tool side ("nice, far from required").
3. **The loss mask is striped:** 1 on tokens the policy sampled (thinking, tool calls, answer, stop tokens), 0 on prompt, template headers and **all tool output**. Masked tokens are still read by the model; they just aren't trained on.
4. **In an RL rollout, step 2 becomes the orchestrator loop and step 3 runs a verifier in the sandbox the model changed.** Steps 4–7 are unchanged except for the striped mask. Keep the sampled token IDs and log-probs; don't re-render text.
5. **Benchmarks are model × harness**, and the details (reasoning continuity, formats, constrained decoding, context management) must be identical between training and inference.
