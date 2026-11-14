# The KL Penalty in RL

Notes on *Regularization in RL, Why RL Generalizes, and Why SFT Forgets | Post-Training Course, Lecture 10* (Nathan Lambert), 0:00–10:53. Covers two YouTube chapters merged into one file: "Intro & the role of regularization" (0:00–2:50, under 3 minutes) and "The KL penalty in RL" (2:50–10:53). Based on `source/lecture-10-regularization-slides.md` and `source/lecture-10-regularization-transcript.md`. Builds on `../lecture-3-policy-gradients/02-wiring-rlhf-for-language-models.md` (the KL "leash") and `../lecture-4-implementing-rl/03-grpo-and-friends-in-code.md` (KL in reward vs. in loss). See `00-map.md` for the full timing diagram.

---

## Where this fits

Lambert calls this "an extra RL lecture" that "could have been in lectures three and four", and "another grab bag of intuitions on how post-training works." Lectures 3–4 introduced the KL penalty as a leash. This lecture asks two bigger questions:

1. **How do RL optimizers change the model's distribution?**
2. **How do we control that change?**

The answer comes in two layers:

| Layer | What it is | File |
|---|---|---|
| **Explicit** regularization | The KL penalty β·KL(π_θ ‖ π_ref): a term you add on purpose | this file |
| **Implicit** regularization | The *shape* of the optimization: RL samples from itself (reverse KL), SFT from data (forward KL) | `02`, `03` |
| Other tools | Extra loss terms, and 2026's shift to masking sampler-vs-trainer drift | `04` |

---

## Slide: "How do these optimizers change the distributions of the models? How do we control it?"

The opening question card. Nothing more on it.

---

## Slide: "Recall: The RLHF process"

The RL step maximizes reward from the reward model **minus a penalty for drifting from the reference model**. "This lecture is about that penalty, and what happens with and without it."

*(The diagram is the standard RLHF overview, reconstructed from the slide caption and transcript.)*

```
prompt ──► policy π_θ (being tuned) ──► completion ──► reward model ──► r_θ
                    │                                                    │
                    └── compared with the frozen reference π_ref ──► − λ·KL
                                                                         ▼
                                                           r = r_θ − λ·KL  ──► RL update
```

This is exactly lecture 3's shaped reward. Connects to: `../lecture-3-policy-gradients/02-wiring-rlhf-for-language-models.md`.

---

## Slide: "RLVR has regularization too, but different best practices"

- Same RL loop, but a **verification function** replaces the reward model (lecture 5).
- "Reasoning models (before tool use) often **dropped the KL penalty** to enhance learning."
- With large-scale tool use, regularization is "coming back into vogue", but **aimed at drift from the *sampling distribution***, not a KL penalty to a reference model. That's `04`.

Lambert: the default β × KL term "came out of vogue in the RLVR era and the tool use era," "but a lot of these regularization ideas still apply, and it's about how to control your model from changing too much during a single RL run."

Connects to: `../lecture-5-reasoning-models/03-common-implementation-patterns.md`, trick #4 (β = 0 for short RLVR runs).

---

## Slide: "This lecture"

The plan:
1. The **KL penalty** that controls RL
2. **The two directions of KL divergences**
3. **Why RL generalizes better than SFT**
4. **Other related work**

---

## Slide: "Aside: Watch lecture 9 first"

Lecture 9 was over-optimization (`../lecture-9-over-optimization/`). Lambert: "regularization and the tools we're talking about are **a solution to lecture nine**." The slide embeds lecture 9's slides inside this deck ("an infinite cascade of slide decks").

---

## Slide: "A KL penalty to the reference model controls reward"

The canonical LLM reward function:

```
r  =  r_θ  −  λ_KL · D_KL( π_RL(y|x) ‖ π_ref(y|x) )
```

Each piece in words:

| Piece | Meaning |
|---|---|
| **r** | The reward the RL algorithm actually sees |
| **r_θ** | The learned reward model's score (θ here = the reward model's weights) |
| **λ_KL** (usually written **β**) | How hard the leash pulls. Bigger = stay closer |
| **π_RL** | The policy being trained |
| **π_ref** | The frozen copy from before RL started, "there have been no updates at all" |
| **D_KL(π_RL ‖ π_ref)** | How far the policy has drifted from that copy |

What the slide adds beyond lecture 3:

- **History.** KL control predates LLMs: Natasha Jaques's work on dialogue agents (2017), then fine-tuning pretrained models from human feedback (2020). "These are essentially the references for where this idea of a KL penalty came from."
- **It's a reverse KL.** "You sample from the policy you're training, and you're scoring it against the reference model." It "punishes the policy for putting mass where the reference would not." (What "reverse" means is the subject of `02`.)
- **"KL distance" is a figure of speech.** Lambert says "KL distance" all the time, meaning "this optimization distance... how far apart the models are." But KL is **not a true distance**, because it isn't symmetric: D(P ‖ Q) ≠ D(Q ‖ P).
- **λ vs. β.** Same thing. The modern literature writes β. (Lambert asks viewers to track down when the naming changed.)

*(My note: a tiny asymmetry check. P = {0.9, 0.1}, Q = {0.5, 0.5}. D(P ‖ Q) = 0.9·log(0.9/0.5) + 0.1·log(0.1/0.5) ≈ 0.53 − 0.16 = 0.37. D(Q ‖ P) = 0.5·log(0.5/0.9) + 0.5·log(0.5/0.1) ≈ −0.29 + 0.80 = 0.51. Different numbers, so it isn't a distance.)*

---

## Slide: "Measuring KL in practice"

The definition becomes an average over samples:

```
D_KL(P ‖ Q) = E_{x ~ P} [ log P(x) − log Q(x) ]
```

In words: **draw samples from P**, and for each one, take "how likely P thinks it is" minus "how likely Q thinks it is" (both in logs). Average.

**Which distribution do you sample from?** The one in the **first slot**. Lambert admits "I always get caught up on this." The rule:

> KL(P ‖ Q): sample from **P**. If P is the model you're training, it's a **reverse** KL.

For the RL penalty, P = π_θ (we already have its samples: they're the completions from step 2) and Q = π_ref. So the KL is nearly free: "it interleaves very nicely with the RL process because you already have most of the metadata."

The slide's code:

```python
tokens = model.generate(inputs)                            # step 2: sample from the POLICY
logprobs     = log_softmax(model.forward(tokens).logits)   # policy's view of those tokens
ref_logprobs = log_softmax(ref_model.forward(tokens).logits)  # step 4: reference's view
token_lp     = gather(logprobs, tokens)                    # pick the log-prob of each actual token
ref_token_lp = gather(ref_logprobs, tokens)
kl_approx = token_lp.sum(-1) - ref_token_lp.sum(-1)        # sequence-level KL estimate
```

It's the same per-token KL_t = log π_θ − log π_ref from lecture 3, summed over the sequence. (Lambert: "I don't love this code that Claude wrote for me here," but it shows the idea. Real libraries do it more efficiently.) For `gather` and `log_softmax`, see `../lecture-4-implementing-rl/00-code-map.md`.

Two practical points:
- **Watch the KL curve.** "A very large KL usually means a bug or a broken model." Spikes and weird behavior are warning signs. (Lecture 4's monitoring table: 0 → 2–5 is healthy, > 10 is trouble.)
- **Lower-variance estimators** k1, k2, k3 exist (covered in the course's Q&A 2, from John Schulman's note on approximating KL). Lambert: "you look at the KL divergence between individual tokens in order to estimate the full shape of that distribution."

### The three estimators (my addition)

The slide only names them. Here they are for one sampled token, with p = policy prob and q = reference prob, and ratio r = q / p:

| Estimator | Formula | Notes |
|---|---|---|
| **k1** | log p − log q  (= −log r) | The one in the code above and in lecture 3. Correct on average, but noisy and **can be negative** for a single token |
| **k2** | ½ (log r)² | Always ≥ 0, lower noise, slightly biased |
| **k3** | (r − 1) − log r | Always ≥ 0, correct on average, low noise. The one GRPO's paper uses |

**Example 1:** policy says 0.6, reference says 0.3. r = 0.5.
- k1 = log 2 ≈ **0.69**. k2 = ½(0.69)² ≈ **0.24**. k3 = (0.5 − 1) + 0.69 ≈ **0.19**.

**Example 2:** policy says 0.3, reference says 0.6. r = 2.
- k1 = log 0.5 ≈ **−0.69** (negative, even though the models disagree!). k3 = (2 − 1) − 0.69 ≈ **0.31** (positive).

Averaged over many samples k1 and k3 give the same answer. k3 just gets there with less noise, because a single token never reports "negative drift."

---

## Slide: "What the curves look like in a real run"

*(Plot reconstructed from the slide caption and transcript.)* An **OLMo 2 7B** GRPO run, RL directly on the **base** model (R1-Zero style, no SFT), math with verifiable rewards, from Ai2's open-instruct with public W&B logs. Over ~1M episodes:

```
reward  │            ___.---‾‾‾‾‾     climbs
        │      __.-‾‾
        │ __.-‾
KL      │                  .-‾‾\_/‾‾\_.-‾   rises from 0, then wanders
        │        __.--‾‾‾‾
        │ 0 ___.-
        └──────────────────────────────────► episodes (~1M)
```

- **This run has β = 0.** No KL penalty at all, but the KL to the reference is **still logged**. You can measure drift without penalizing it.
- It starts at 0, because at step 1 the policy *is* the reference.
- Lambert: "a weird looking curve, but it is fine. As long as the shape looks fine and the numbers are not too big." The slide: "This is a healthy shape."

*(My note on timing: with β = 0, step 4 (the reference forward pass) isn't needed for training. You skip it, and can even drop the reference model from memory. To draw a KL curve like this one you still need reference log-probs, so a run that logs KL at β = 0 is doing step 4 just for monitoring.)*

Connects to: `../lecture-5-reasoning-models/03-common-implementation-patterns.md` trick #4 (β = 0 for short RLVR), and `../lecture-5-reasoning-models/01-introduction-and-core-concepts.md` ("Zero" runs from a base model).

---

## Slides: "Static or dynamic KL penalties? β began as a feedback controller" (two slides)

The first RLHF-on-LMs paper (**Ziegler et al., 2019, "Fine-Tuning Language Models from Human Preferences"**, "probably the earliest modern RLHF paper") **did not fix β**. It picked a **target KL** and let a controller chase it:

```
e_t     = clip( (KL_now − KL_target) / KL_target ,  −0.2, +0.2 )
β_next  = β_now × (1 + K_β · e_t)          with K_β = 0.1
```

In words:
- **e_t** = "how far off target are we, as a fraction of the target," capped at ±20% so one weird batch can't swing β wildly.
- If KL is **too high**, e_t > 0, so **β grows** and the leash tightens.
- If KL is **too low**, e_t < 0, so **β shrinks** and the policy is freed up.

They call it "a log-space proportional controller." Lambert: "it's kind of a mouthful, but essentially they're just trying to maintain a KL divergence target."

**Worked example (my addition):** target KL = 6, β = 0.1, K_β = 0.1.

| Batch's measured KL | (KL − 6)/6 | e_t after clip | New β |
|---|---|---|---|
| 9 | +0.5 | +0.2 | 0.1 × 1.02 = **0.102** (tighter) |
| 6.6 | +0.1 | +0.1 | 0.1 × 1.01 = **0.101** |
| 3 | −0.5 | −0.2 | 0.1 × 0.98 = **0.098** (looser) |

Each nudge is small. Over hundreds of batches it steers the run onto the target.

**Why bother:** "runs with different seeds land on the *same* KL budget, making experiments comparable." You compare runs at equal drift, not equal β.

**When it runs** (my reading of the formula and the original code): once per batch, **between batches**. Measure this batch's mean KL, update β, and the new β shapes the **next** batch's rewards. It never changes anything inside the current step.

```
batch t:  steps 1–7 with β_t  ──► measure mean KL of batch t ──► β_{t+1} = β_t(1 + K_β·e_t)
batch t+1: steps 1–7 with β_{t+1} ...
```

**The idea is older than RLHF:**
- The original **PPO paper** had an **adaptive KL penalty** variant that doubled or halved β around a target.
- Constrained RL later used full **PID controllers** on the penalty multiplier (Stooke et al., 2020).

**Modern practice swung back to simple:** a small **static** β, or in many RLVR recipes **no KL term at all**. The code links on the second slide tell the story:

| Code | Year | What it does |
|---|---|---|
| OpenAI `lm-human-preferences` | 2019 | The original adaptive controller |
| TRL's `AdaptiveKLController` | v0.11.4 | Adaptive, then **deleted in the modern rewrite** |
| Ai2 open-instruct (today) | – | **Static `beta = 0.05`**, applied **directly in the loss** (GRPO style) |

---

## When is the KL computed? (the timing view)

This is the question that matters most for following later chapters. It depends on **where** the KL goes (lecture 4 `03`):

```
                            PPO (KL in REWARD)                  GRPO (KL in LOSS)
──────────────────────────────────────────────────────────────────────────────────────────
step 2  generate            save policy log-probs (π_old)       same
step 4  reference pass      ref log-probs, CACHED               ref log-probs, CACHED
step 5  advantages          KL_t = π_old logp − ref logp        –
                            r_t = −β·KL_t (+ R at last token)
                            → GAE → advantages FIXED
step 6  loss (×K inner)     KL already baked in; no KL term     KL_t = NEW logp − ref logp
                                                                RECOMPUTED every inner step;
                                                                + β·KL_t in the loss
between batches             (adaptive β update, if used)        same
──────────────────────────────────────────────────────────────────────────────────────────
```

| Question | PPO-style | GRPO-style |
|---|---|---|
| Is the reference's view cached? | Yes, step 4 | Yes, step 4 |
| Which policy log-prob goes into KL? | The one **at generation** (π_old) | The **current** one, inside the inner loop |
| Does KL change during the inner loop? | No, it's frozen inside the advantages | Yes, the policy moves, so the KL moves |
| Does gradient flow through the KL? | Only indirectly, via the advantage weights | **Directly**: it's a loss term |
| With β = 0? | Skip step 4 (unless logging) | Skip step 4 (unless logging) |

And across the whole run, the KL is always against the **same frozen π_ref**, so it measures **total drift since training started**, not drift since the last update (lecture 3 `00`).

---

## The whole section in five lines

1. **Regularization is the answer to lecture 9's over-optimization**: control how far one RL run moves the model.
2. **The explicit tool is r = r_θ − β·KL(π_θ ‖ π_ref)**, an idea from Jaques's pre-LLM dialogue work. It's a **reverse KL**: sample from the policy, score against the reference.
3. **"KL distance" is loose talk.** KL isn't symmetric, so it isn't a true distance, but it's a useful "how much optimization have we spent" gauge. Watch its curve: it should rise smoothly from 0, never spike.
4. **β started as a feedback controller** (Ziegler 2019: chase a target KL, nudge β each batch). Modern practice: a small static β (open-instruct: 0.05) or **β = 0** in RLVR, while still logging KL.
5. **Timing:** reference log-probs are cached at rollout (step 4). PPO folds KL into the rewards before advantages. GRPO recomputes KL from the current policy at every loss step.
