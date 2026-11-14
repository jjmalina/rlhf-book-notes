# Part 1: The Eras of Post-Training Evaluation

Notes on *How Evaluation Has Evolved with Frontier Model Progress | Post-Training Course, Lecture 12* (Nathan Lambert), 03:39–17:36. Based on `source/lecture-12-evaluation-slides.md` and `source/lecture-12-evaluation-transcript.md`. Follows on from `01-introduction-frontier-evaluation-is-harder-than-ever.md`. See `00-map.md` for the eras timeline table.

---

## Where this fits

This is the history part: **for each kind of model, how was it prompted, how was it graded, and which benchmarks did people care about?** The main thread is that every change in *how models are trained* forced a change in *how they're tested*.

**When does this happen?** Everything in this file happens **after a checkpoint exists**: you take frozen weights, prompt them, grade the outputs, and get a number. None of it touches steps 1–7 of the training loop. The pieces that change from era to era are the **prompt**, the **sampling settings** and the **grader**.

```
era by era, only these three boxes change:

 [ PROMPT ] ──► frozen checkpoint ──► [ SAMPLING / SCORING ] ──► [ GRADER ] ──► score
 few-shot →                          log-likelihood →            letter match →
 zero-shot →                         greedy generation →         regex / "The answer is" →
 zero-shot CoT →                     temperature > 0, n samples  LLM-as-a-judge →
 harness + sandbox (03)                                          tests run in a sandbox (03)
```

---

## Slide: "Benchmarks mirror the training goals of their era"

> "The key to understanding evals: popular benchmarks are a **reflection of the training best practices of their moment**."

| Era | Years | What training was trying to improve | So benchmarks tested... |
|---|---|---|---|
| **Chat** | 2022–23 | Basic knowledge and chat style | Internet-scale knowledge (MMLU), chat quality |
| **Multi-skill** | 2023–24 | Everything at once: math, code, factuality, safety | One benchmark per skill |
| **Reasoning & tools** | 2024–26 | Hard math, code, puzzles, solved with long chains of thought | Expert-written hard problems |
| **Agents & real work** | now | End-to-end knowledge work inside products | Software, finance, legal tasks in harnesses |

Lambert on the multi-skill era: once people saw models "can work generally, it was this mad rush to get better at everything at once." Then those benchmarks saturated and attention specialized. Today: "so many of the evals that we cared about in the last five years are just fully saturated, but what we're doing now is really open-ended and much harder to measure."

---

## Slide: "Base models (before post-training): few-shot prompting"

A base model is an **autocomplete** engine. Ask it a bare question and it doesn't know what kind of text should come next. It might continue with another question, or ramble.

The fix was **few-shot prompting**: put 3–8+ worked examples in the prompt so the model continues the *pattern*. The slide's example (abbreviated):

```
### Example 1
Q: A right triangle has legs of lengths 3 and 4. What is the length of its hypotenuse?
(A) 5  (B) 6  (C) 7  (D) 8
Correct Answer: (A)

### Example 2
Q: Which of the following is the chemical symbol for Sodium?
(A) Na  (B) S  (C) N  (D) Ca
Correct Answer: (A)

### Now answer the new question in the same style:
Q: Which theorem states that if f is continuous on [a,b], it attains a max and a min?
(A) Mean Value Theorem (B) Intermediate Value Theorem (C) Extreme Value Theorem (D) Rolle's Theorem
Correct Answer:
```

After seeing two examples, the most natural next text is ` (C)`. The examples teach the **format** (answer with a letter), not the knowledge.

The canonical evals were named by the number of examples: **5-shot MMLU**, **8-shot GSM8K**.

---

## Slide: "Grading: log-likelihood vs. exact match"

Two ways to get an answer out of the model at the `Correct Answer:` position:

| | **Log-likelihood scoring** | **Generation + exact match** |
|---|---|---|
| What you do | Don't sample. Look at the probability the model gives each option (just the letter `A`/`B`/`C`/`D`, or the full answer string) and take the highest | **Sample** a completion, then extract the answer and compare it to the key |
| Deterministic? | Yes | No (if temperature > 0) |
| Mirrors real use? | No: users never see probabilities | Yes |
| Standard for | **Pretraining** evals, where models couldn't always produce a clean format | **Post-training**, since about 2024 |
| What can go wrong | Little | **Answer extraction** bugs, which "only became more complex with agentic models today" |

**A tiny log-likelihood example (my addition).** At `Correct Answer: (`, the model's next-token probabilities might be:

| Token | Probability |
|---|---|
| `C` | 0.61 |
| `B` | 0.22 |
| `A` | 0.09 |
| `D` | 0.05 |
| (every other token) | 0.03 |

The highest of the four options is `C`, which is correct, so the score is 1. It never had to write anything, so formatting can't go wrong. That's also why it only works for multiple choice.

Generation lets you sample several times and aggregate: **majority voting**, or **pass@k**. Lambert: "pass@k is a very important fundamental to know if you're doing evaluation."

*(My note on timing: this is the same "one forward pass, read off the log-prob" operation as step 4 of the lecture 3 loop (reference log-probs), and generation is step 2. Evaluation reuses the same machinery, just on frozen weights and held-out questions.)*

---

## Slide: "The math behind pass@k"

Connects to: `../lecture-5-reasoning-models/01-introduction-and-core-concepts.md` ("Key terms"), which has the plain-language version of the formula and the n = 4, c = 1 worked example. This slide adds **why** the formula is built that way and **how sampling settings change the result**.

**pass@k = the probability that at least one of k samples solves the problem.**

### Two tempting shortcuts, and why both are wrong

1. **Generate exactly k, report whether any passed.** This is an honest measurement, but for each problem it's a single coin flip (0 or 1), so it's **high variance**.
2. **Estimate the success rate p̂ = c/n from a few samples, then plug it into 1 − (1 − p̂)^k.** This is **biased**: on average it gives the wrong answer, even with lots of problems.

**Why the plug-in is biased: a worked example (my addition).** Suppose one problem has a true success rate p = 0.5. The true pass@2 is 1 − 0.5² = **0.75**. Draw n = 2 samples and see what each estimator says:

| Outcome (c correct of 2) | Chance | Plug-in 1 − (1 − c/2)² | Codex estimator 1 − C(2−c, 2)/C(2, 2) |
|---|---|---|---|
| c = 0 | 0.25 | 0 | 0 |
| c = 1 | 0.50 | 1 − 0.25 = 0.75 | 1 − 0/1 = 1 |
| c = 2 | 0.25 | 1 | 1 |
| **Average** | | 0.375 + 0.25 = **0.625** (too low) | 0.5 + 0.25 = **0.75** ✓ |

The plug-in comes out too low on average. The Codex formula is right on average ("unbiased").

### The fix, from the Codex paper (OpenAI, 2021)

Sample **n ≥ k** completions per problem, count the **c** that pass, and average over problems:

```
pass@k = average over problems of [ 1 − C(n−c, k) / C(n, k) ]
```

- **C(n−c, k) / C(n, k)** is the chance that k draws (without replacement) from your n samples are **all failures**. One minus that is "at least one success."
- **Larger n gives a tighter estimate** at the same k. The Codex paper used **n = 200 for k ≤ 100**.
- So a reported pass@k "is normally with a higher n than the k, and it is using this formula under the hood."

### The knobs interact: temperature

> "**Higher temperature can hurt pass@1 but help pass@100.** So a reported 'pass@1' depends on n and the sampling settings, not just the model."

**Example (my addition).** One hard problem:

| Setting | Chance one sample is right | pass@1 | pass@100 |
|---|---|---|---|
| Temperature 0 (greedy): always the same answer, which happens to be wrong | 0 | 0 | **0** (100 copies of the same wrong answer) |
| Temperature 0.8: answers vary, right 10% of the time | 0.10 | 0.10 | 1 − 0.9¹⁰⁰ ≈ **1.0** |

Greedy decoding gives you one shot repeated. Sampling gives you real diversity, which is what pass@k rewards. That's the same diversity GRPO needs to get non-zero advantages (lecture 3 `05`), and why RL rollouts in step 2 are sampled, not greedy.

---

## Slide: "The early pipeline was simple"

*(The diagram isn't in the folder. Reconstructed from the transcript. Slide credit: Florian Brand.)*

```
prompt ──► base LLM ──► completion ──► grader ──► score (maybe pass@k / maj@k)
   ▲
   └── almost everything that could go wrong was HERE
```

> "The same base LLM with different prompts and different ways of telling the model how to format the answer for the same evaluation could meaningfully change the score."

People kept **specialized prompts for every evaluation**. Compare this with the agentic pipeline in `03`, where the prompt is one of many boxes.

---

## Slide: "Chain of thought (CoT) emerged to enable progress on harder problems"

Chain of thought came **before** ChatGPT-style post-training, while people were still prompting base models (Wei et al. 2022; Kojima et al. 2022).

The idea: make the few-shot examples show **intermediate steps**, so the model copies the habit of working things out before answering.

| Standard prompting | Chain-of-thought prompting |
|---|---|
| Q: Roger has 5 tennis balls. He buys 2 more cans of 3. How many now? **A: The answer is 11.** | Same Q. **A: Roger started with 5 balls. 2 cans of 3 tennis balls each is 6 tennis balls. 5 + 6 = 11. The answer is 11.** |

Math and reasoning scores jumped.

The phrase **"The answer is"** matters for grading: it's a marker that tells the extractor where the answer starts. "If you just get a big bunch of text, you have to be able to extract which one of them is the answer."

As models got chattier, people dropped the long examples and just appended **"Let's think step by step."** Lambert: today it's more like "think very hard, spend your effort. There's all sorts of, like, fake bribing to get the models to work harder."

*(My note: reasoning models (lecture 5) are what you get when you **train** this behavior in with RL instead of **prompting** for it. The `<think>` block is a trained chain of thought.)*

---

## Slide: "Entering the chat era: zero-shot instruction following"

**Instruction tuning** (FLAN, T0) and then RLHF taught models to **answer questions directly**. The input became:

```
User: "What is the capital of France?"
Assistant:
```

No examples, so this is **zero-shot**. Lambert: "In retrospect, this was a huge deal for the applicability of language models... but this was new."

Three consequences for evaluation:

1. **LLM-as-a-judge** appeared because answers were open-ended, and "evals imitated RLHF training" (a learned judge grading answers, like a reward model).
2. Canonical chat evals: **MT-Bench**, **AlpacaEval**, and community-scale **Chatbot Arena** (people vote between two anonymous models).
3. **Multiple-choice evals like MMLU stayed**, but in flux: now answered zero-shot, sampling the answer letter at **temperature 0**.

Connects to: `../lecture-5-reasoning-models/02-the-2025-cambrian-explosion.md`, where Lambert says these chat evals were "so many ways to game" that early open RLHF results came with "a question mark." It's one reason verifiable math/code evals were so welcome.

---

## Slide: "The emergence of zero-shot prompting took time!"

Lambert's own example: **Tülu 3** (trained summer 2024, released fall 2024) used a **mix**: few-shot for knowledge tasks, zero-shot CoT for math and code. The skills it was evaluated on:

| Skill | Benchmarks |
|---|---|
| Knowledge | MMLU, PopQA, TruthfulQA |
| Reasoning | BigBenchHard, DROP |
| Math & code | MATH, GSM8K, HumanEval |
| Instruction following & safety | IFEval, others |

**By early 2025 everyone used zero-shot.** Today, few-shot prompting a post-trained model is rare, unless you're deliberately testing in-context learning.

The MMLU example question on the slide (GDP per capita of the US in 1850, adjusted to 2011 prices: about $300 / $3k / $8k / $15k) shows what MMLU really tests: "Internet trivia more than intelligence — but it tracked pretraining knowledge well."

---

## Slides: "Encouraging the models to reason" (two slides)

### Per-benchmark prompts

Tülu 3 wasn't a reasoning model, so to get it to reason a bit *and* produce an extractable answer on MMLU, the team wrote a custom prompt:

```
Answer the following multiple-choice question by giving the correct answer letter in parentheses.
Provide CONCISE reasoning for the answer, and make sure to finish the response with
"Therefore, the answer is (ANSWER_LETTER)" where (ANSWER_LETTER) is one of (A), (B), (C), (D), (E), etc.

Question: {question}
(A) {choice_A}
...
Answer the above question and REMEMBER to finish your response with the exact phrase
"Therefore, the answer is (ANSWER_LETTER)" ...
```

Note that the format instruction appears **twice**. That's what it took to get reliable extraction.

The payoff: the gap between few-shot multiple-choice MMLU and zero-shot CoT MMLU "could be, like, **eight points**." Same model, same questions. Just the eval setup.

Then o1 arrived (September 2024, the first reasoning model) and "really pushed everybody to finish changing their prompting to be this chain of thought way, because all the models needed time to reason."

### Sampling settings joined the prompt as part of the eval

Reasoning models need **temperature > 0** for their best scores. Qwen's model cards literally say **"DO NOT use greedy decoding."** The recommended settings are in the model's `generation_config.json`, and using them is "free" performance.

Lambert: getting this right "takes a lot of resources and engineer time... to make sure that you are getting the evaluation settings to match the original paper."

*(My note, practical: when you compare models for your coding agent, read each model's recommended temperature/top-p and use them. Running a reasoning model greedy because "temperature 0 is more reproducible" can quietly cost several points, and it can make one model look worse than another for reasons that have nothing to do with ability. (Greedy reasoning models can also get stuck repeating themselves.))*

---

## Slide: "Formatting became fragile as usage became more open-ended"

- **Formatting mismatches can take a model from 60% to near 0.** "It is far easier to lose performance with a prompt than to gain it." Lambert's rule of thumb: "if you see one of your evaluation scores drop just dramatically to zero from, like, 50%... normally, it's a formatting issue."
- **Answer extraction is brittle**: rigid suffixes ("The answer is:") or regexes hunting for the answer anywhere in the text.
- **Training formats can conflict.** NuminaMath answers with `\boxed{42}`. MetaMath answers with `The answer is: 42`. **Training on both can be worse than either alone**, because the model doesn't know which format to use.
- **Format-agnostic grading takes a lot of effort**, so it was rare. **LLM judges became popular even just as answer extractors**, because "LLMs are pretty good at being able to read what the proposed answer was."

**A concrete extraction failure (my addition).** The key is `42`. Three model outputs:

| Model output ends with... | Regex `The answer is: (\d+)` | `\boxed{(\d+)}` extractor | LLM extractor |
|---|---|---|---|
| `The answer is: 42` | 42 ✓ | none ✗ | 42 ✓ |
| `$\boxed{42}$` | none ✗ | 42 ✓ | 42 ✓ |
| `So the answer is 42.` | none ✗ (no colon) | none ✗ | 42 ✓ |

The model was right all three times, but the score depends on which extractor you picked. If you trained with one format and evaluate with another, you can see "60% → near 0."

Connects to: `../lecture-5-reasoning-models/03-common-implementation-patterns.md` #8 (format rewards). That's the *training-side* fix: reward the model for always producing an extractable format, so the step-3 verifier and the eval grader can both find the answer.

---

## Slides: "Reasoning & tool-use pushed the industry to harder tasks" (two slides)

Reasoning models **saturated the old suites**, so new ones appeared:

| Area | Benchmarks |
|---|---|
| Knowledge | **GPQA Diamond**, **Humanity's Last Exam** (HLE), **FrontierMath** |
| Math | Recent **AIME** contests (new each year, so less likely to be in training data) |
| Software | **SWE-bench** (+ variants), **LiveCodeBench** |

**Question writing moved from the internet to grad students, PhDs and professors.** Writing questions became expert labor. Lambert: these are "really, really labor-intensive, and they tend to have a lot of little bugs in them."

### The HLE example: when the answer key is wrong

> **Q:** What was the rarest noble gas on Earth as a percentage of all terrestrial matter in 2002?
> **Official answer:** Oganesson

According to the slide, the official answer is wrong three ways:
- oganesson is **not a gas** (it's predicted to be a solid)
- it's **not noble** (it's predicted to be reactive)
- it's **not terrestrial** (it's synthetic: about 5 atoms ever made, first synthesized in 2002)

Lambert's honest caveat: "I didn't double check this. Claude found this. So if this is a mistake by Claude, let me know."

The lesson: **past a certain difficulty, verifying the answer key is the bottleneck.** "Expert-written no longer means correct." Mislabeled items are common, and **most evals saturate at 90–95%, not 100%**. "None of the benchmarks will ever get to 100% because they all have labeling errors."

*(My note: this matters for RLVR too. If a benchmark's training split has wrong labels and you use it as a verifier in step 3, RL rewards the model for matching the wrong answer. A mislabeled eval just costs you a few points of ceiling. A mislabeled training verifier actively teaches the wrong thing.)*

---

## Slide: "Today: evals of real work"

- The frontier evals are **end-to-end professional tasks**: **SWE-bench Verified**, **Terminal-Bench** (computer/terminal tasks), **GDPVal** (OpenAI), **APEX** (Mercor).
- **Task authors are experienced professionals**: GDPVal's experts average **14 years** of industry experience. APEX's average **7+ years** at firms like Goldman Sachs and McKinsey. "Expert task-writing is the new cost center."
- **The models aren't evaluated bare.** They run **inside harnesses and products**: "you evaluate the latest Claude model in Claude Code. You evaluate the latest OpenAI model inside of Codex CLI." That's last lecture's subject (`../lecture-11-tools-and-agents/`) and Part 2 of this one (`03`).

---

## Slide: "Every era ends the same way: saturation"

*(The Epoch AI plot isn't in the folder. Reconstructed from the slide caption and transcript.)*

The plot shows many major benchmarks, each as a curve of the best model's score over time. Each one starts low, climbs, and flattens near its ceiling. Lambert: there's "a slower era to begin with," but zoomed out, benchmarks fall "so fast."

> "Benchmarks are consumable. As scores approach the ceiling, only the hardest (and mislabeled) items remain, and the benchmark stops separating models."

**Why saturation kills a benchmark (my addition).** Take a 500-question benchmark at 95%. Of the 25 questions left, say 15 have wrong answer keys. Only 10 real questions remain, so two models at 95.4% and 96.2% differ by about 4 questions. That's within the noise you'll see in `04`.

Lambert's takeaway: "The assumption should be that benchmarks will keep getting solved, and we always need to make more."

---

## The whole section in six lines

1. **Benchmarks mirror the training goals of their era:** chat → multi-skill → reasoning & tools → agents and real work. **Every era ends in saturation.**
2. **Base models needed few-shot prompts** (to teach the format) and were graded by **log-likelihood** (deterministic, no formatting risk). Post-trained models are prompted **zero-shot** and graded by **generation + answer extraction**.
3. **pass@k** = chance at least one of k samples is right. Estimate it with the **Codex formula** using n ≥ k samples, because the naive version is noisy and the plug-in version is biased. **Temperature changes the answer**: higher temperature can hurt pass@1 but help pass@100.
4. **Chain of thought** started as a few-shot prompting trick ("Let's think step by step") before it was trained in. Reasoning models then made **temperature > 0** part of the eval ("DO NOT use greedy decoding").
5. **Formatting is the classic silent killer:** a mismatch can take a model from 60% to ~0. Extraction is brittle, training formats conflict (`\boxed{}` vs. "The answer is"), and LLM extractors became common.
6. **Each new era needs more expert labor per question** (PhDs → professionals with 7–14 years' experience), and answer keys become the bottleneck: most evals top out at 90–95% because of mislabeled items.
