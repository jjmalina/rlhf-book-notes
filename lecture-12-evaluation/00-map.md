# Lecture 12 Map

A one-page map for Lecture 12: **where evaluation happens** (outside the training loop), **how an eval differs from a training reward**, **a timeline of the eval eras and benchmarks**, and a glossary. It's the lecture 12 companion to `../lecture-3-policy-gradients/00-training-loop-timeline.md`.

Covers: `01`–`04` (the whole lecture). `04` also covers the short "Takeaways & conclusion" chapter (30:41–32:10).

---

## The one idea

> **A benchmark score is not a property of the model. It's a property of the whole system: prompt, sampling settings, inference engine, harness, sandbox, hardware, timeout and grader.** Benchmarks also mirror the training goals of their era, and every era ends in saturation.

---

## When does evaluation happen? (Outside the training loop)

Lectures 3–5 were all about the inner loop: steps 1–7, repeated thousands of times. **Evaluation never runs inside steps 1–7.** It runs *between* checkpoints, *after* a run, and *after* release. Nothing it produces goes into a gradient.

```
                    THE WHOLE POST-TRAINING PIPELINE (lecture 5 00-map)

  pretrain ──► mid-train ──► SFT ──► RL (RLHF / RLVR) ──► distill ──► release
     │            │           │          │                   │           │
     ▼            ▼           ▼          ▼                   ▼           ▼
   [EVAL]       [EVAL]      [EVAL]     [EVAL]              [EVAL]      [EVAL]   ← after each stage:
                                                                               "is this checkpoint better?"

  ZOOM IN ON ONE RL RUN (lecture 3 timeline, steps 1–7):

  BEFORE THE RUN
    • decontaminate training data against eval sets (n-gram search, 04)
    • eval the starting checkpoint = the baseline every later number is compared to
    • (lecture 5 #1 difficulty filter uses the same "sample N times + check" machinery,
       but on TRAINING prompts, so it's data prep, not evaluation)

  DURING THE RUN
    ┌────────────── inner loop: thousands of times ─────────────────┐
    │ 1 sample prompts → 2 generate → 3 SCORE (verifier/RM) → 4 ref  │
    │ → 5 advantages → 6 loss → 7 optimizer step ★ weights change ★ │
    └───────────────────────────┬───────────────────────────────────┘
                                │ every ~N steps: save checkpoint
                                ▼
                     ┌──────────────────────────┐
                     │ EVAL JOB (separate GPUs, │  held-out prompts, eval sampling settings,
                     │ often async)             │  many samples (avg@k). Output: a number on a
                     └────────────┬─────────────┘  dashboard. NO gradient. Policy unchanged.
                                  ▼
                       a human (or script) reads it

  AFTER THE RUN
    • pick which checkpoint ships; compare runs / data mixes / hyperparameters
    • run the (expensive) full agentic suite; report a public subset

  THE SLOW OUTER LOOP (this is what "hillclimbing" means)
    eval numbers ──► researcher changes data mix / hyperparams / reward ──► new run ──► eval ...
    One "step" of this loop takes days to weeks. The researcher is the optimizer.

  AFTER RELEASE
    • Arena / human A/B tests / third-party reruns / users finding things "through use" (01)
```

*(This diagram is my addition, assembled from Lambert's statements that "every training decision... is made off benchmark numbers" and that the real job is "statistical power deciding between model weights.")*

---

## Training reward vs. evaluation score (even with the same checker)

In RLVR you might literally run the same function, say `run_unit_tests(patch)` or `extracted_answer == 77`, as both the step-3 reward and the eval grader. They still do different jobs:

| | **Training reward** (step 3) | **Evaluation score** |
|---|---|---|
| When | Every step, inside the loop | Between checkpoints / after training / after release |
| On which prompts | Training prompts (filtered to the 20–80% band, lecture 5 `03`) | **Held-out** benchmark prompts, never trained on (ideally) |
| Sampling settings | Whatever helps exploration (often temperature ~1.0, G samples per prompt) | The model's **recommended** settings (e.g. Qwen's "DO NOT use greedy decoding"), fixed and documented |
| Per-sample output | 0/1 or a score → **advantage → gradient** | 0/1 → **averaged into one number** |
| Who consumes it | The optimizer (automatically, step 7) | A human making a decision |
| Changes the weights? | Yes, directly | **No**, only indirectly, through the decisions people make |
| Harness / environment | Training environment | Ideally the *same* environment code (Prime Intellect's verifiers, `04`), but with held-out tasks |
| Failure mode | **Reward hacking**: the policy exploits the checker (lecture 3 `02`, `../lecture-9-over-optimization/`) | **Eval gaming / contamination / noise**: the number goes up for reasons other than capability (`04`) |

*(My note: the training reward curve going up is a training-set number. It tells you the policy is learning to satisfy the checker on prompts it has seen. Only the eval tells you whether that carried over to new problems. That's why lecture 4 `06` says "watch held-out eval scores, and actually read sample outputs.")*

**The catch Lambert raises (`04`):** "Evals are more of a training target than a monitoring lens." If you pick checkpoints and data mixes by an eval, the outer loop is optimizing for it. Do that enough times and the eval drifts towards being a training signal too: Goodhart's law, one level up.

---

## Timeline of eval eras and benchmarks

| Era | Years | Model type | How it was prompted | How it was graded | Canonical benchmarks | File |
|---|---|---|---|---|---|---|
| **Base models** | pre-2022 | Autocomplete (GPT-3) | **Few-shot**: 3–8+ worked examples in the prompt | **Log-likelihood** over answer options (deterministic) | 5-shot MMLU, 8-shot GSM8K | `02` |
| (CoT appears) | 2022 | Base models | Few-shot examples *with reasoning*, then "Let's think step by step" | Extract the answer after "The answer is" | GSM8K, math | `02` |
| **Chat** | 2022–23 | Instruction-tuned / RLHF (FLAN, T0, ChatGPT) | **Zero-shot** chat: `User: ... Assistant:` | **LLM-as-a-judge**; MCQ sampled at temperature 0 | MT-Bench, AlpacaEval, Chatbot Arena, MMLU | `02` |
| **Multi-skill** | 2023–24 | General post-trained models (Tülu 3) | Mix of few-shot (knowledge) and zero-shot CoT (math, code) | Generation + exact match, per-benchmark prompts | MMLU, PopQA, TruthfulQA, BBH, DROP, MATH, GSM8K, HumanEval, IFEval | `02` |
| **Reasoning & tools** | 2024–26 | Reasoning models (o1 →) | Zero-shot, long CoT. **Temperature > 0 required** | Exact match / extraction, LLM extractors, pass@k | GPQA Diamond, Humanity's Last Exam, FrontierMath, AIME, SWE-bench, LiveCodeBench | `02` |
| **Agents & real work** | now | Agents in harnesses (Claude Code, Codex CLI) | A **harness** in a **sandbox**, hours-long | Regex or LLM judge on the final state | SWE-bench Verified, Terminal-Bench, GDPVal, APEX; METR time horizons | `02`, `03` |

Each row ends the same way: **saturation** (Epoch AI's plot, `02`). Each new row needs more expert labor per question: internet trivia → PhDs → professionals with ~7–14 years of experience.

---

## How this connects to lectures 3–5

| Lecture 12 idea | Builds on | Where |
|---|---|---|
| Evals run outside steps 1–7 | The training-loop timeline | `../lecture-3-policy-gradients/00-training-loop-timeline.md` |
| Same checker as reward and grader, different jobs | RLVR verifier = step 3 | `../lecture-5-reasoning-models/01-introduction-and-core-concepts.md` |
| pass@k unbiased estimator, temperature trade-off | pass@1 / pass@k, the headroom gap | `../lecture-5-reasoning-models/01-introduction-and-core-concepts.md` ("Key terms") |
| avg@k to beat noise | Bigger batches reduce gradient noise: the same "average more samples" logic ("it's an estimate, like a poll") | `../lecture-4-implementing-rl/02-ppo-implementation.md` |
| Inference engine bugs change scores | Inference vs. trainer log-prob mismatch (FP32 LM head) | `../lecture-5-reasoning-models/02-the-2025-cambrian-explosion.md` |
| Format bugs → near-0 scores; `\boxed{}` vs. "The answer is" | Format rewards (#8) | `../lecture-5-reasoning-models/03-common-implementation-patterns.md` |
| Agents game evals (git history, hardcoded tests) | Reward hacking / over-optimization | `../lecture-3-policy-gradients/02-wiring-rlhf-for-language-models.md`, `../lecture-9-over-optimization/` |
| Random-reward RL "improves" Qwen → contamination | The discover-vs-elicit debate | `../lecture-5-reasoning-models/04-open-questions-and-conclusion.md` |
| Test-time scaling confounds lab comparisons | Inference-time scaling, majority voting (#13) | `../lecture-5-reasoning-models/03-common-implementation-patterns.md` |
| Harnesses, tool calls, sandboxes | Tool use and agents | `../lecture-11-tools-and-agents/` |
| Human A/B, Elo, Arena | Preference data | `../lecture-8-preference-data/` |

---

## Glossary (new in lecture 12)

| Term | Meaning |
|---|---|
| **Benchmark / eval** | A fixed set of tasks plus a grading rule. Produces one number per model |
| **Few-shot prompting** | Putting k worked examples in the prompt so a base model copies the format ("5-shot MMLU") |
| **Zero-shot prompting** | Just asking the question, no examples. Works once a model is instruction-tuned |
| **Chain of thought (CoT)** | The model writes intermediate steps before the answer |
| **Log-likelihood scoring** | No sampling: compare the probability the model gives each option (A/B/C/D) and pick the highest. Deterministic |
| **Generation + exact match** | Sample a full answer, extract the final answer, compare it to the key |
| **Answer extraction** | The code (regex, "The answer is", `\boxed{}`, or an LLM) that pulls the final answer out of free text |
| **LLM-as-a-judge** | Another LLM grades open-ended answers (or just extracts the answer) |
| **pass@k** | Chance at least one of k samples is right. Estimated with n ≥ k samples (`02`) |
| **avg@k** (mean@k) | Run k samples per problem and **average** their pass/fail. Same expected value as pass@1, much less noise (`04`) |
| **maj@k** | Majority voting: the most common of k answers is the model's answer |
| **Temperature / top-p** | Sampling knobs. Temperature 0 = greedy (always the most likely token) |
| **Saturation** | Scores near the ceiling (usually 90–95%, because of mislabeled items), so the benchmark no longer separates models |
| **Harness** | The software loop around an agent model: system prompt, tool definitions, context management |
| **Sandbox** | A reproducible container (files, tools, rules) the agent acts in |
| **Inference engine** | The serving software (vLLM, SGLang, an in-house engine) that runs the model and parses tool calls |
| **Time horizon** (METR) | The length of task (in human-expert time) a model completes 50% of the time. Doubling roughly every 7 months |
| **σ (eval variance)** | Standard deviation of a score across reruns of the same model. Typically 0.25–1.5 points |
| **Hillclimbing** | Repeatedly changing training to push a specific eval up |
| **Held-out / train / dev / test** | Classical ML splits. A held-out test set is one you never tune on |
| **Contamination** | Eval test items (or near-copies) in the training data, so the score measures memorization |
| **Decontamination** | Searching training data for eval items (n-gram / substring / fuzzy match) and removing them |
| **n-gram** | A run of n consecutive words/tokens. Two texts sharing many 13-grams are probably copies |
| **Perturbed benchmark** | The same problems rewritten with new numbers/wording, to catch memorization |
| **Private test set** | Never released, so it can't leak (e.g. Scale's leaderboards) |
| **Eval gaming / reward hacking in evals** | The agent finds a shortcut (future git commit, hardcoded outputs) instead of solving the task |
| **Pareto frontier (cost vs. intelligence)** | Models where nothing else is both cheaper *and* smarter. Lambert: "the most important plot in evaluation today" |
| **Statistical power** | How reliably an eval can tell two models apart, given its noise |
