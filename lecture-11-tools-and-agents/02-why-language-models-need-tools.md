# Part 1: Why Language Models Need Tools (+ Related Work)

Notes on *How Language Models Use Tools and the Path to Agents | Post-Training Course, Lecture 11* (Nathan Lambert), 07:29–15:30. Based on `source/lecture-11-tools-and-agents-slides.md` and `source/lecture-11-tools-and-agents-transcript.md`. Follows on from `01-introduction-and-motivations.md`.

---

## Where this fits

`01` said *why* tools exist (cutoff, can't act). This section pins down the **vocabulary**, shows what a tool call looks like inside one answer, and walks the **history** from 2015 to 2026. It ends with how tool use is **evaluated**. The mechanics (who executes what, which tokens are trained on) come in `03`.

---

## Slide: "Three terms people have conflated"

From most general to most specific:

| Term | Definition | Example |
|---|---|---|
| **Tool use** | The general strategy: the model emits a **structured request** (tool name + arguments), an **orchestrator** executes it, the **results are appended** to the context, and the model **continues generating** | `search("current US president")` |
| **Function calling** | A specific *format* of tool use: arguments must match a **declared schema** (usually JSON Schema), so they can be parsed and validated reliably | `{"name": "get_weather", "arguments": {"location": "NYC"}}` |
| **Code execution** | A special case where the tool is a **code interpreter**, "the most general tool of all." Code goes in, execution output comes out | `<code>print(2**100)</code>` → `<output>1267650600228229401496703205376</output>` |

Lambert's history note on function calling: early on there were "constrained JSON-only models" because models weren't reliable at the format. "Today, they all follow the tool format perfectly all the time," so the distinction comes up less.

Why code execution matters so much: it gives the model "a **deterministic** form of thinking," which balances the "structural limitation that models otherwise are probabilistic" (they sample from a distribution).

---

## Slide: "Escaping probabilistic generation: an example tool-use task"

**Task:** print π to 50 digits, without reciting it from memory and risking hallucination.

The slide shows this inside the model's generation:

```text
... the model is mid-generation, sampling tokens ...

<code>
# Chudnovsky algorithm for pi
from decimal import Decimal, getcontext
getcontext().prec = 60
...
print(str(C / S)[:52])
</code>

<output>
3.14159265358979323846264338327950288419716939937510
</output>

... the output is now in context; generation continues ...
```

**The model writes the program; the interpreter provides the answer.** Tools "let models remove some of the stochastic elements of 'stochastic parrots'."

What to notice:
- **It all happens inside one response** to the user. The tool call isn't a separate conversation turn from the user's point of view.
- **Two authors in one text stream:** everything in `<code>` was **sampled by the model**; everything in `<output>` was **written by the Python interpreter**. That split is the whole of `03`'s masking story.
- Lambert's fun question: which open-weight model, with no tool, gets the most digits of π right? Many have probably memorized a few dozen digits, but "they'll still break down at some point." With a tool, you get any number of digits.

*(My note: this is the same reason RLVR (lecture 5) works on math: a deterministic checker is more trustworthy than a probabilistic model. Here the deterministic program sits **inside** the answer instead of after it.)*

---

## Slide: "A short history of models using tools"

| Year (first arXiv) | Work | What it added |
|---|---|---|
| 2015–2020 | Neural Programmer-Interpreters; **retrieval augmentation (RAG)** | Handing a program off to something else to run; pulling outside documents into the model. RAG was "one of the first most popular forms of tool use." Superseded by general search for consumer LLMs, but still very popular in business |
| 2021 | **WebGPT** (OpenAI) | Trained a model to **browse the web** with RLHF. "Super ahead of its time," and one of the few papers that detailed RLHF at scale |
| 2022 | TALM, **PAL**, **ReAct** | TALM bootstraps tool-use data; PAL offloads computation to Python; **ReAct interleaves reasoning and actions**: "the one to know" |
| 2023 | **Toolformer**, Gorilla (1,645 APIs), ToolLLM (16,000+ APIs); OpenAI ships **function calling** in the API and **Code Interpreter** in ChatGPT | Function calling takes off after ChatGPT. Synthetic tool-use data. Scale in number of APIs |
| 2024 | **MCP** (Model Context Protocol) | Some standardization of the tool side (`03`) |
| 2025 | OpenAI **o3** | **Multistep tool calls inside its reasoning.** Search + code execution + reasoning, combined with large-scale RL. Lambert: "one of my favorite models that I've ever tried" |
| 2026 | Terminal and coding agents | "Become the frontier of post-training" |

The slide's footnote: the years are when work first appeared on arXiv. Citation years show official publication, so some lag a year.

Lambert: this "is generally how long it took to lay the foundations," and "future innovations will come much faster."

---

## Slide: "ReAct: Reasoning and acting are one generation"

The ReAct quote (paper posted Oct 2022, ICLR 2023):

> *"...reasoning traces help the model induce, track, and update action plans as well as handle exceptions, while actions allow it to interface with and gather additional information from external sources such as knowledge bases or environments."*

**Before ReAct**, chain-of-thought reasoning and tool-calling actions were separate research areas. **ReAct put them in one token stream**:

```
think → act (tool call) → observe (tool output) → think → act → observe → ... → answer
```

"This is really what the modern tools look like." o3's tool calls inside its thinking are "this idea, scaled up with RL." Lambert's term for today's version: **tool-integrated reasoning**.

Connects to: `../lecture-5-reasoning-models/01-introduction-and-core-concepts.md` (the `<think>` block). ReAct-style agents are reasoning models whose thinking is interrupted by real observations.

---

## Slide: "Toolformer: Models 'teach themselves' tools"

Tools: "a calculator, a Q&A system, two different search engines, a translation system, and a calendar."

A self-labelling (synthetic data) pipeline:

1. **Prompt** the model to insert candidate API calls into its own pretraining text.
2. **Execute** the calls.
3. **Keep only the calls whose results reduce perplexity** on the text that follows.
4. **Fine-tune** on the filtered corpus.

No human demonstrations needed, so it's "an early instance of synthetic-data flywheels" (compare `../lecture-7-synthetic-data/`).

**Perplexity in plain terms:** how surprised the model is by the next tokens. Lower = better predicted. Lambert: "you want perplexity to be lower if you're training an accurate language model."

**Worked example of step 3** *(my addition)*. Text: *"Out of 1,400 participants, 400 (or 29%) passed the test."*

| Version | Model's surprise at "29%" | Keep the call? |
|---|---|---|
| No call | High: it has to guess the percentage | – |
| With `[Calculator(400/1400) → 0.29]` inserted before "29%" | Low: the answer is right there in context | ✅ yes, the tool result **helped** predict the following text |
| With `[Calendar() → Today is Friday]` inserted | Unchanged: irrelevant to "29%" | ❌ no, a useless call |

So the filter keeps tool calls only where they actually help, which teaches the model *when* calling a tool is worth it.

*(The slide's figure of Toolformer's example predictions, where the model decides on its own to call QA, calculator, translator and Wikipedia search mid-text, isn't in the sources. Reconstructed from the caption.)*

---

## Slide: "How tool use is evaluated"

| Level | Benchmark | What it checks |
|---|---|---|
| **Schema / format** | Berkeley Function Calling Leaderboard (built on Gorilla's APIBench) | Exact match on tool name and arguments; valid JSON. "Can models successfully call tools at all?" |
| **Breadth** | ToolLLM / ToolBench | 16,000+ real-world APIs |
| **Reliability** | **τ-bench** (TauBench) | **pass^k**: succeed on **all** k trials |
| **End-to-end** | **Terminal-Bench** (2.0 is "the most popular evaluation") | Agents do real tasks in containers, checked by verification tests. Long-horizon |

**The eval ladder mirrors the capability ladder:** format → selection → consistency → full tasks.

### pass^k vs. pass@k *(my addition, connecting to lecture 5)*

Lecture 5 `01` defined **pass@k** = the chance that **at least one** of k tries succeeds (coverage). τ-bench's **pass^k** is the opposite: the chance that **all** k tries succeed (reliability).

Example: an agent that solves a task 80% of the time, with tries assumed independent:

| k | pass@k (any of k) | pass^k (all of k) |
|---|---|---|
| 1 | 0.80 | 0.80 |
| 2 | 1 − 0.2² = 0.96 | 0.8² = 0.64 |
| 4 | 1 − 0.2⁴ ≈ 1.00 | 0.8⁴ ≈ **0.41** |
| 8 | ≈ 1.00 | 0.8⁸ ≈ **0.17** |

pass@k goes up with k; pass^k goes down. For an agent a business deploys to handle every customer, pass^k is the number that matters: "80% per try" means it fails at least once in 4 tries more often than not.

---

## The whole section in four lines

1. **Tool use** (general) ⊃ **function calling** (schema-checked format) ⊃ **code execution** (the interpreter as tool, the most general one). Code gives a probabilistic model a deterministic way to compute.
2. **A tool call lives inside one response**: model-written tokens (the call) and tool-written tokens (the output) share one stream.
3. **History:** RAG/WebGPT → ReAct (reason + act in one stream, "the one to know") → Toolformer and function-calling APIs (2023) → MCP (2024) → o3, tools inside reasoning (2025) → coding/terminal agents (2026).
4. **Evals climbed the same ladder:** format (BFCL) → breadth (ToolBench) → reliability (τ-bench's pass^k) → full container tasks (Terminal-Bench).
