# Lecture 10 Map

A one-page map for Lecture 10: **where regularization sits in post-training**, **exactly when each KL-related number is computed in the training loop**, and **the two directions of KL** that the whole lecture turns on. It's the lecture 10 companion to `../lecture-3-policy-gradients/00-training-loop-timeline.md` and `../lecture-4-implementing-rl/00-code-map.md`.

Covers: `01`–`04` (the whole lecture). The 0:00 intro chapter (under 3 minutes) is merged into `01`.

---

## The one idea

> **Two kinds of "staying close to the original model."** The **explicit** one is the KL penalty (β × KL to π_ref), a dial you set. The **implicit** one comes from **where the training samples come from**. RL learns from its **own** samples (reverse KL, mode-seeking), so even with β = 0 it tends to find solutions close to where it started. SFT learns from **someone else's** fixed data (forward KL, mass-covering), so it gets dragged toward that data and **forgets more**.

---

## Where this lecture sits in post-training

```
PRETRAINING ──► SFT ───────────────────────────► RL (RLHF / RLVR) ─────────────► (deploy)
                │                                  │
                │ forward KL to the DATA           │ reverse KL: samples from ITSELF
                │ (imitate fixed text)             │ + optional explicit penalty β·KL(π_θ ‖ π_ref)
                │ → can drift far, forgets more    │ → biased toward small drift ("RL's razor")
                ▼                                  ▼
          Lecture 9 problem: over-optimization of a learned reward
          Lecture 10 answer: regularization (explicit KL, implicit on-policy, other tools)
```

---

## When does each KL-related thing happen? (lecture 3's step numbers)

```
                                       PPO-style (KL in reward)        GRPO-style (KL in loss)
────────────────────────────────────────────────────────────────────────────────────────────────
before the run    copy SFT model → π_θ and π_ref (frozen forever). KL = 0 at step 1.
1. sample prompts                      –                               –
2. generate with π_θ, save log-probs   (= π_old)                       (= π_old)
3. score → R                           –                               –
4. ref log-probs (frozen π_ref,        CACHED for the batch            CACHED for the batch
   one forward pass, no gradients)     (skipped entirely if β = 0)     (skipped entirely if β = 0)
5. advantages                          KL_t = old − ref, folded into   KL NOT here
                                       reward: r_t −= β·KL_t → GAE
                                       (KL now FIXED inside advantages)
6. loss (every inner step)             no KL term                      KL_t = NEW − ref, recomputed
                                                                       every step: + β·KL_t
7. optimizer step                      ★ policy changes ★              ★ policy changes ★
between batches   (2019-style adaptive β only) measure the batch's mean KL → nudge β up or down
                  for the NEXT batch (01).   Logging: the KL curve is plotted here, even if β = 0.
────────────────────────────────────────────────────────────────────────────────────────────────
```

The **implicit** regularization (`02`, `03`) isn't a step at all. It's a consequence of **step 2**: the data the loss sees was written by the policy itself.

---

## Forward vs. reverse KL on one card

| | **Forward KL** D(target ‖ model) | **Reverse KL** D(model ‖ target) |
|---|---|---|
| Who writes the samples | the **target** (dataset, teacher) | the **model being trained** |
| Who uses it | **SFT**, offline distillation | **RL** (with β > 0), on-policy distillation, the KL penalty itself |
| Punished for | missing anywhere the target has mass | putting mass where the target has little |
| Nickname | **mass-covering** ("blanket everything") | **mode-seeking** ("go to a peak, ignore the rest") |
| Lecture 10 claim | **forgets more** | **forgets less**, generalizes better |

Mnemonic: in KL(P ‖ Q) you **sample from whatever is in the first slot**. Model in the first slot = reverse.

---

## How this connects to lectures 3–5

| Lecture 10 idea | Builds on | Where |
|---|---|---|
| KL penalty r = r_θ − β·KL | The "leash", per-token KL_t = log π_θ − log π_ref | `../lecture-3-policy-gradients/02-wiring-rlhf-for-language-models.md` |
| Which model is frozen, which changes | π_ref vs. π_old; the step-by-step loop | `../lecture-3-policy-gradients/00-training-loop-timeline.md` |
| KL in reward (PPO) vs. in loss (GRPO) | Timing: rollout vs. every training step | `../lecture-4-implementing-rl/03-grpo-and-friends-in-code.md`, `00-code-map.md` |
| Watching the KL curve | Monitoring table: KL 0 → 2–5 healthy, > 10 bad | `../lecture-4-implementing-rl/06-loss-aggregation-strategies.md` |
| β = 0 in reasoning RL | Recipe trick #4: KL dropped for short RLVR | `../lecture-5-reasoning-models/03-common-implementation-patterns.md` |
| "RL is on-policy, SFT = advantage 1" | RL loss = −advantage × log π(**own** token) | `../lecture-3-policy-gradients/00-training-loop-timeline.md`, `../lecture-4-implementing-rl/00-code-map.md` |
| Sampler vs. trainer drift (DPPO, IcePop) | Ratio ρ, clipping, FP32 LM head mismatch | `../lecture-3-policy-gradients/04-ppo.md`, `../lecture-5-reasoning-models/02-the-2025-cambrian-explosion.md` |
| Reward-tilted reference π* | The DPO derivation (lecture 6) | `../lecture-6-dpo/` |
| Forward vs. reverse in distillation | On-policy distillation (lecture 7) | `../lecture-7-synthetic-data/` |
| Why regularize at all | Over-optimization (lecture 9) | `../lecture-9-over-optimization/` |

---

## Files

| File | Chapter | Time |
|---|---|---|
| `01-the-kl-penalty-in-rl.md` | Intro + the explicit KL penalty, measuring it, adaptive β | 0:00–10:53 |
| `02-rl-as-a-reverse-kl-loss.md` | RL = reverse KL to a reward-tilted reference; SFT = forward KL | 10:53–21:09 |
| `03-why-rl-generalizes-better-than-sft.md` | Three papers: SFT memorizes, Retaining by Doing, RL's Razor | 21:09–25:15 |
| `04-other-regularization-tools.md` | PPO-ptx, DPO+NLL, margin loss, 2026: no KL, DPPO; summary | 25:15–31:33 |

---

## Glossary additions (new in lecture 10)

| Term | Meaning |
|---|---|
| **Regularization** | Anything that stops training from changing the model too much or in bad ways |
| **Explicit regularization** | A term you add on purpose, e.g. β × KL to π_ref |
| **Implicit regularization** | Lambert's name (book ch. 15) for staying-close that comes from the *shape* of the optimization, e.g. on-policy sampling |
| **KL divergence** D(P ‖ Q) | Average of log P(x) − log Q(x) over samples **drawn from P**. 0 when equal. Not symmetric, so not a true distance |
| **"KL distance"** | Lambert's colloquial name: "how much optimization distance you've spent" |
| **Forward KL** | D(target ‖ model). Samples from the target. What SFT minimizes |
| **Reverse KL** | D(model ‖ target). Samples from the model. What the KL penalty and KL-regularized RL are |
| **Mass-covering** | Forward KL's behavior: spread probability to cover everything the target might do |
| **Mode-seeking** | Reverse KL's behavior: concentrate on a high-probability (high-reward) region, ignore the rest |
| **Reward-tilted reference** π* | π_ref with each answer's probability multiplied by exp(r/β), renormalized. The best possible policy for the KL-regularized objective |
| **Partition function** Z(x) | The renormalizing sum in π*. A constant per prompt |
| **Support** | The set of answers a distribution gives non-zero probability. π* can't leave π_ref's support |
| **Entropy** H | How spread out a distribution is. In the SFT = forward KL proof, the data's entropy is a constant |
| **Adaptive KL controller** | Change β each batch to keep measured KL near a **target KL** (Ziegler et al. 2019, PPO's adaptive variant) |
| **k1 / k2 / k3** | Three per-sample KL estimators with different noise (Schulman's note on approximating KL) |
| **OOD** | Out-of-distribution: test problems that differ from training (e.g. a rule shift) |
| **Catastrophic forgetting** | Losing old abilities while learning a new task |
| **RL's Razor** | Among equally good solutions, RL finds the one with the smallest KL from the base model |
| **PPO-ptx** | InstructGPT: add pretraining next-token log-likelihood to the RL objective |
| **Margin loss** | Llama 2 reward-model loss that demands a bigger score gap when annotators rated a bigger difference |
| **Trust region** | A limit on how far one update can move the policy (clipping, KL, masking) |
| **Total variation (TV)** | Half the sum of absolute probability differences. Bounded between 0 and 1 |
| **DPPO** | Divergence Proximal Policy Optimization: mask tokens whose estimated TV between sampler and trainer is too large |
| **IcePop** | GLM's method: mask tokens whose train/inference probabilities disagree too much |
| **Masking vs. clipping** | Masking drops a token from the gradient outright. Clipping caps how much it can push |
