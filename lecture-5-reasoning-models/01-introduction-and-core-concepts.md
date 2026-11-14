# Introduction & Core Concepts

Notes on *The Rise of Reasoning Models | Post-Training Course Lecture 5* (Nathan Lambert), 00:00–10:41. Based on `source/lecture-5-reasoning-models-slides.md` and `source/lecture-5-reasoning-models-transcript.md`. Builds on `../lecture-3-policy-gradients/` (the math) and `../lecture-4-implementing-rl/` (the code). See `00-map.md` for the model timeline and how every recipe trick maps onto the training loop.

---

## Where this fits

Lectures 3–4 were **how** RL on LLMs works. Lecture 5 is **what people actually built with it** in the 2025 "reasoning model" wave, and which tricks stuck. It's mostly history and practice, with very little new math.

Lambert frames it as documenting "a very fun time in the world," so that "when people are looking back in a decade" there's a record of how we got here. He recorded it around **June 2, 2026**, after leaving Ai2 (the Allen Institute for AI).

The one idea to hold onto:

> **Reasoning models are the lecture 3–4 algorithms (mostly GRPO-style) applied with *verifiable* rewards (RLVR) on math and code. That combination teaches models to "think" in long chains before answering, and then it spread to agents and tool use.**

---

## Slide: "From RL to reasoning"

- Lectures 3–4: the core policy-gradient implementation (PPO, GRPO, loss aggregation, async training).
- This lecture: the rise of reasoning models, techniques specific to reasoning, and a recap of 2025's core models.

---

## Slide: "The LeCun cake"

At NeurIPS 2016, Yann LeCun described machine learning as a cake:

| Layer | LeCun's 2016 metaphor | What it turned out to be for LLMs |
|---|---|---|
| **The cake** (the bulk) | Unsupervised / predictive learning | **Pretraining**: self-supervised next-token prediction on internet text ("training on data that creates its own labels") |
| **The icing** | Supervised learning | **SFT / mid-training**: training on curated examples with correct answers |
| **The cherry on top** | Reinforcement learning | **RL**: RLHF, then RLVR |

Lambert: "it's just so remarkable that... Yann's vision kind of came true for the most important AI technologies that have existed." (He's made "cherry RL" merch for his newsletter.)

**The twist on the slide:** with agents, RL may become **more than just the cherry in compute**, possibly more compute than pretraining one day. But the *order* of the layers still describes how the technology was built: you need the cake before the cherry works.

The connection to lecture 3: this is why RL on LLMs starts from a strong model and uses a **reference model / KL** to stay near it. The cherry sits on a cake that's already baked.

---

## Slide: "RLHF vs RLVR: How reward changes everything"

**The algorithms are the same. The reward is different, and that changes everything about how you build models.**

| | RLHF | RLVR |
|---|---|---|
| **Example prompt** | *"Explain opportunity cost in economics."* | *"What is the sum of all primes < 20?"* / *"Write `fib(n)`"* |
| **How it's scored** | A learned **reward model** judges clarity, accuracy, completeness. These are learned preferences, "no definitive answer" | A **verifier** checks: `extracted_answer == 77` → 1. `assert fib(10) == 55` passes → 1 |
| **Can it be gamed?** | Yes, it's a learned proxy you can **over-optimize** (lecture 3 `02`; lecture 9 `03`) | Much harder: "a much more precise tool," so "you could optimize the models much further" |

Where the RLHF column's reward model comes from (preference pairs → Bradley–Terry), and a fuller RLHF vs. RLVR comparison: `../lecture-8-preference-data/00-map.md`. RLVR can still be hacked (empty outputs, test-file edits): `../lecture-9-over-optimization/03-signatures-of-over-optimization-and-misalignment.md`.

The new intuition RLVR opened: **"we could scale all this training on problems where we already know the answer."** Math and code have huge supplies of problems with checkable answers, and that was the takeoff point.

**The caveat on the slide:** "the clean distinction here is pedagogical, but in practice rare." Real post-training mixes RLHF, RLVR, **rubrics** (graders that check a list of criteria, a "softly verifiable" middle ground) and tools. RLVR was the **door** to reasoning models and agents.

---

## Slide: "The RLVR feedback loop"

*(The diagram isn't in the folder. This is from the caption and the transcript.)*

```
prompt ──► policy (LLM) ──► completion ──► VERIFIER (instead of a reward model) ──► reward 0/1
              ▲                                                                       │
              └───────────────────── RL update (GRPO/PPO) ◄──────────────────────────┘
```

It's the same loop as lecture 3's timeline, with the reward model swapped for a **verification function**. In lecture 4 `00-code-map.md`, that's step 3: `rewards = verifier(...)`.

Lambert's forward-looking point: the verifier "is becoming an **environment** again, which is closer to the classical notion of RL." With tool use and agents, the model acts in a real environment (runs code, searches, calls tools) over many steps, not just one answer and one check. That brings back the classical RL setting from lecture 3's "Classical RL vs. RLHF" slide. "The power of RL will become more promising even if the implementation is harder."

What that looks like at the token level (multi-turn rollouts, tool output masked out of the loss, sandboxes as the verifier): `../lecture-11-tools-and-agents/03-infra-how-tool-calls-work.md` and `04`.

---

## Slide: "Recall: RL training vs inference-time scaling" (the OpenAI o1 plots)

Two different "more compute = better" curves, both roughly **linear on a log scale** (each 10× more compute buys a similar step up):

| | **Train-time compute** (left plot) | **Test-time compute** (right plot) |
|---|---|---|
| What you scale | How long the **RL training run** goes | How many tokens the model **thinks for at answer time** |
| When | During training | Every time a user asks something |
| Name | RL scaling | **Inference-time scaling** |

Lambert calls inference-time scaling "the fundamental property of AI now": performance grows with the log of the number of thinking tokens.

**How they're linked:** long RL runs are what **teach** a model to use long chains of thought productively, which **unlocks** test-time scaling. "The early era of reasoning models is more defined by this RL training, which you could just do these big RL runs which unlock inference-time scaling." This lecture focuses on the training side.

---

## Slide: "Recall: What thinking tokens look like"

**A standard model** answers directly:

> The primes less than 20 are 2, 3, 5, 7, 11, 13, 17, 19. Their sum is **77**.

**A reasoning model** first writes inside special `<think>` tokens:

> `<think>` I need to find all primes less than 20. Starting from 2: 2 is prime. 3 is prime. 4 = 2×2, not prime. 5 is prime... Let me add them up: 2+3=5, 5+5=10, ... 58+19=77. `</think>`
>
> The answer is **77**.
>
> **Verification**: `extracted_answer == 77` → Reward = 1

What to notice:
- **The reward only checks the final answer.** Nothing grades the thinking directly. RL discovers that thinking helps get answers right, and reinforces whatever thinking patterns tend to precede correct answers. That's lecture 3's "advantage copied to every token": every thinking token in a correct completion gets pushed up.
- **Thinking can be thousands of tokens** on hard problems. That's why lecture 4 `06` (loss aggregation and length bias) and the inference bottleneck (lecture 4 `04`) matter so much for reasoning models.
- Lambert on what real traces look like: "very odd. It's very repetitive. Sometimes they do guess and check, but it helps the models get smarter." Closed models (OpenAI, Claude) hide them; open-weight models show them.
- The answer must be in an **extractable format** (here, boxed) so the verifier can find it. Hence "format rewards" in `03`.

---

## Slide: "Key terms for this lecture"

### pass@1 and pass@k (evaluation metrics)

| Metric | Question it answers | Measures |
|---|---|---|
| **pass@1** | If the model gets **one** try, how often is it right? | Reliability |
| **pass@k** | If the model gets **k** tries, how often is **at least one** right? | **Coverage**: can it solve the problem *at all*? |

pass@k always goes up as k grows (more chances).

**How it's estimated:** generate **n** attempts (n ≥ k), and count **c** correct. Then:

```
pass@k ≈ 1 − C(n−c, k) / C(n, k)
```

Lambert admits he "gets stuck on" this one ("everybody has... one equation in their field where they kinda get stuck on it"). Here's the plain version:

> **pass@k = 1 − (the chance that k attempts picked at random from your n are *all wrong*).**
> C(n−c, k) counts the ways to pick k attempts using only wrong ones. C(n, k) counts all the ways to pick k.

**Example:** n = 4 attempts, c = 1 correct.
- pass@1 = 1 − C(3,1)/C(4,1) = 1 − 3/4 = **0.25** (just c/n)
- pass@2 = 1 − C(3,2)/C(4,2) = 1 − 3/6 = **0.50** (half of all pairs contain the correct one)
- pass@4 = 1 − C(3,4)/C(4,4) = 1 − 0/1 = **1.0** (all 4 together always include it)

Why you sample n > k and use the formula, rather than just trying k times: it averages over all possible groups of k, which is much less noisy.

**Why this matters for RL:** the **gap between pass@k and pass@1** is headroom. If the model *can* solve a problem in 1 of 8 tries, RL can learn to make that success more reliable. It's also exactly the setup GRPO needs (lecture 3 `05`): a prompt where some of the G samples succeed and some fail, which gives non-zero advantages. It comes back in `03` (test-time scaling) and `04` (the "discover vs. elicit" debate).

More on pass@k in later lectures:
- **pass^k** (all k tries right, the reliability you want from an agent) vs. pass@k: `../lecture-11-tools-and-agents/02-why-language-models-need-tools.md`.
- Why the simple plug-in estimate is biased, how temperature changes pass@k, and avg@k: `../lecture-12-evaluation/02-the-eras-of-post-training-evaluation.md` and `04`.

### Algorithm names you'll see

- **DAPO**: an early, influential RLVR paper that modified GRPO and released open data. Its changes, **relaxed/asymmetric clipping** and **per-token loss aggregation** (lecture 4 `06`), are widely copied. *(Note: the slide expands DAPO as "Dynamic Advantage Policy Optimization." As far as I know, the paper's own expansion is "**D**ecoupled Clip and Dynamic s**A**mpling **P**olicy **O**ptimization," which names its two main ideas.)*
- **CISPO**: from the MiniMax-M1 paper. It caps the importance-sampling weight instead of switching tokens off (lecture 3 `05`, lecture 4 `03`). Lambert: "very lasting in 2026, continuing to grow in relevance."

---

## The whole section in four lines

1. **Reasoning models = lecture 3–4 RL + verifiable rewards (RLVR) on math/code**, which teaches models to think in long `<think>` chains before answering.
2. **RLHF vs. RLVR:** the same algorithms, different reward. A verifier is precise and hard to game, so you can push optimization much further. In practice everything is mixed (RLHF + RLVR + rubrics + tools), and the verifier is turning back into a full **environment**.
3. **Two scaling curves:** longer RL training (train-time) is what unlocks longer, more useful thinking at answer time (test-time / inference-time scaling). Both improve roughly with log(compute).
4. **pass@1** = reliability, **pass@k** = coverage (1 − chance that k tries are all wrong). The gap is RL's headroom, and it's also where GRPO gets non-zero advantages.
