# GRPO and Modern Variants

Notes on *Understanding Policy Gradient Algorithms for RL on LLMs | Post-Training Course Lecture 3* (Nathan Lambert), 50:04–57:35 (end of lecture, including the conclusion). A slide-by-slide breakdown, based on `source/lecture-3-policy-gradients-slides.md` and `source/lecture-3-policy-gradients-transcript.md`. Follows on from `04-ppo.md`. See `00-training-loop-timeline.md` for where everything happens.

---

## Where this fits

GRPO is built entirely from pieces you already have:

> **GRPO = PPO's clipping + RLOO's "compare against the group" baseline − PPO's critic**

```
from RLOO (03):  sample G completions per prompt, judge each against the group
from PPO  (04):  ratio ρ + clipping, so the batch can be reused safely
dropped:         the value model (critic) and GAE
```

Lambert calls it "a breath of fresh air" after PPO: "it's like PPO, but a bit simpler."

The section also covers two 2025 variants, **GSPO** and **CISPO**. Both tweak how the ratio and clipping work, to fix instability when training very large models.

---

## Slide: "GRPO: PPO for the reasoning era"

**GRPO** (Group Relative Policy Optimization) was introduced in the **DeepSeekMath** paper (2024) for math reasoning. **DeepSeek R1** made it famous, and it has since become the default algorithm for RL on language models.

**Core idea:** generate **G** completions per prompt and use the group's reward statistics as the baseline, so no learned critic is needed. (G is the same as RLOO's K, just a different letter.)

Why it took over:
- **Simpler** to implement and debug than PPO.
- **Saves memory:** one fewer LLM-sized model (no critic).
- **A natural fit for RLVR:** verifiers give clear signals like 0/1 for correct/incorrect, and comparing within a group of attempts works well with that.
- Lambert's broader point: "we have seen many times again in AI that [being simpler] can be a reason for something to be very successful."

---

## Slide: "GRPO advantage"

For the G completions of one prompt:

```
advantage_i = (R_i − mean of the group's rewards) / (std of the group's rewards)
```

This is a **z-score**, which you may know from statistics: how many standard deviations above or below the group average this completion is.

It has two parts:
1. **Subtract the mean:** "better or worse than the group?" This is the same baseline idea as RLOO, except GRPO includes the completion itself in the mean.
2. **Divide by the std (spread):** puts every prompt on the **same scale**, whether its rewards are spread widely or bunched together.

### Worked examples (RLVR, reward = 1 if correct, 0 if wrong, G = 4)

**A hard prompt: 1 of 4 correct.** Rewards [1, 0, 0, 0], mean = 0.25, std ≈ 0.43

| Completion | Reward | Advantage |
|---|---|---|
| correct | 1 | **+1.73** (a rare success, strongly reinforced) |
| wrong ×3 | 0 | −0.58 each (mild push down) |

**An easy prompt: 3 of 4 correct.** Rewards [1, 1, 1, 0], mean = 0.75, std ≈ 0.43

| Completion | Reward | Advantage |
|---|---|---|
| correct ×3 | 1 | +0.58 each (mild push up) |
| wrong | 0 | **−1.73** (a rare failure, strongly suppressed) |

This is the "was it better than expected *for this prompt*?" idea from `01` and `03`: the surprising outcome gets the big push.

### The shortcoming Lambert points out: all the same means no signal

**Rewards [1, 1, 1, 1]** (too easy) or **[0, 0, 0, 0]** (too hard): every answer equals the mean, so **every advantage is 0, and there's no gradient from this prompt.**

The model learns nothing from prompts it always gets right or always gets wrong. It only learns where its answers **disagree**. Lambert: "when there is diversity in the completions it works very well." *(Implementation detail: the std is 0 here, so code adds a tiny number to avoid dividing by zero.)*

### Same advantage for every token

As with RLOO, **each token in completion i gets the same advantage**. There's no per-token credit like PPO's critic gives. GRPO accepts cruder credit assignment in exchange for dropping the critic.

---

## Slide: "GRPO objective" *(the equation looks scary, but it's all familiar parts)*

```
J(θ) = average over the G completions of
         average over that completion's tokens of
           [ PPO-clipped( ρ_{i,t} × A_i )  −  β × KL_{i,t} ]
```

Piece by piece:

| Piece | What it is | Seen before in |
|---|---|---|
| Average over G completions | Combine the group | `01` (averaging into one gradient) |
| Average over each completion's tokens (1/\|a_i\|) | Normalize by length so long answers don't dominate | new (more in Lecture 4) |
| min(ρ × A, clip(ρ) × A) | PPO's clipped objective, exactly | `04` |
| **ρ per token, A per completion** | Each token gets its own ratio, but they all share the completion's advantage | `04` (ratio), `03` (shared advantage) |
| −β × KL | The KL penalty against π_ref | `02`, but **moved** (below) |

### The KL penalty moved from the reward into the loss

| | PPO (`02`, `04`) | GRPO (default) |
|---|---|---|
| **Where KL goes** | Subtracted from the **reward** before advantages | A separate term in the **loss** |
| **Effect** | Drift makes an answer look *worse* (lower advantage) | Advantages come purely from the rewards. KL pulls each token back toward π_ref on its own. |
| **Timing** | Step 5 (computing rewards/advantages) | Step 6 (the loss) |

The job is the same either way: keep the model close to where it started. Only where it's plugged in changes. It's also **optional**. Many RLVR setups drop it (`02`), which removes the reference model too.

---

## Slide: "GRPO vs PPO"

| | PPO | GRPO |
|---|---|---|
| **Value function** | Learned critic V_φ | None |
| **Advantage** | Per-token, via GAE | Per-completion, group z-score |
| **KL penalty** | In the reward | In the loss (default, optional) |
| **Models in memory** | 4 (policy, value, reference, reward model) | 3 (policy, reference, reward model), or **2** without KL |
| **Complexity** | Higher | Lower |
| **Popular for** | General RLHF | Reasoning / RLVR (DeepSeek R1) |

The slide's one-liner: **"GRPO is PPO minus the value function, with a statistical baseline instead."**

The "2 without KL" case is the policy plus the reward model. In RLVR the reward model is often a verifier *program* rather than an LLM, so the setup can shrink to **one LLM plus a checker**. That's a big memory saving compared with PPO's four LLMs.

---

## Slide: "RLOO vs GRPO"

They're very close relatives. Both sample several completions per prompt and compare each against its peers.

| | RLOO (`03`) | GRPO |
|---|---|---|
| **Baseline** | Mean of the **other** K−1 | Mean of the **whole group** (including itself), then ÷ std |
| **Update style** | REINFORCE: one step, no clipping | PPO-style: ratio + clipping |
| **KL penalty** | Optional, in the reward | Optional, in the loss |

The slide's key line: **without the std division, GRPO's advantage equals RLOO's up to a scaling constant.** You saw this in `03`: the plain group mean gave advantages that were exactly 3/4 of RLOO's for K = 4. So the difference in baselines is mostly cosmetic. The real differences are the std scaling and the PPO-style clipping.

Lambert notes RLOO uses leave-one-out "to make sure there's not bias," and points to a section in the RLHF book (rlhfbook.com) with the math linking the two.

---

## Slide: "GSPO: Sequence-level ratios" *(2025, from the Qwen team)*

### The problem

In PPO and GRPO, **each token has its own ratio ρ**, and each token is clipped on its own. In long answers, this gets fragile:
- **A single token can swing a lot**, hit the clip limit immediately, and lose its gradient, even when the answer as a whole barely changed.
- Or one token with a huge ratio can **dominate** the update.

Lambert says this is especially bad in **mixture-of-experts (MoE)** models. An MoE model contains many sub-networks ("experts"), and a router picks which experts handle each token. A small weight update can flip the router's choice for a token, which makes that token's probability jump. So individual token ratios swing much more than in a regular model.

### The fix

Use **one ratio per completion** instead of one per token: the **geometric mean** of the token ratios (the average of the log-ratios, exponentiated). You don't need the formula. What it does:

> **Clip the whole answer based on how much the answer as a whole changed, not token by token.**

**Example**, ε = 0.2 (band 0.8–1.2), a 4-token answer with token ratios [1.0, 1.0, 1.0, 1.6]:

| | What happens |
|---|---|
| **GRPO (per-token)** | Token 4 (1.6) is past 1.2, so it's clipped and loses its gradient. Tokens 1–3 continue. |
| **GSPO (per-completion)** | Geometric mean ≈ **1.125**, inside the band, so the whole answer keeps learning. |

Lambert: it "adds a bit more of a buffer before clipping kicks in." Because it's an average, the ratio stays in a sensible range regardless of answer length, so the same ε works for short and long answers. He says it has "proven to be decent."

---

## Slide: "CISPO: Clipped importance sampling" *(2025, from the MiniMax-M1 paper)*

### The idea

In PPO/GRPO, clipping acts as an **off switch**: past the band, a token's gradient becomes **zero** (`04`). CISPO asks: what if clipping never turned anything off?

CISPO **caps the ratio's weight instead of switching the token off**:

| | Past the band, the token's gradient is… |
|---|---|
| **PPO / GRPO** | **Zero.** The token stops learning for this batch. |
| **CISPO** | **Still there**, but its weight is capped at the band's edge (e.g. 1.2) |

So **every token always gets some gradient**. The ratio can only amplify it by a limited amount.

### Reading the formula without the calculus

```
J_CISPO = Σ  sg(clipped ρ) × A × log π_θ(token)
```

- **A × log π_θ(token)** is plain REINFORCE from `01` and `03`: advantage × log-prob.
- **clipped ρ** is the ratio, capped to the band.
- **sg(…)** means "**stop-gradient**": treat this number as a fixed constant and don't backprop through it. The capped ratio is just a multiplier on the REINFORCE term, not something the optimizer pushes on.

So CISPO is **REINFORCE with a capped importance weight**. That's why Lambert sums it up as: "GSPO is an importance-sampling modification to GRPO, and CISPO is more of an importance-sampling modification to something like REINFORCE."

### Asymmetric bounds

CISPO uses a **bigger upper limit than lower limit** (ε⁺ > ε⁻). Good tokens are allowed to rise further than bad tokens fall, which encourages the model to **explore** promising new behavior. (This is the "different ε for top and bottom" preview from `04`.)

It's built to help models learn "in high-instability environments," and Lambert says it "has been validated as a very useful algorithm."

---

## Slide: "GRPO architecture"

*(The diagram isn't in the folder. This is from the caption and the transcript.)*

```
prompt ──► policy ──► G completions ──┬──► reward model / verifier ──► G rewards ──► group z-score ──► G advantages
                                      │                                                                    │
                                      └──► reference ──► per-token KL (optional) ──────────────────────────┤
                                                                                                           ▼
                            policy ◄── clipped update steps (per-token ratio vs. saved π_old log-probs) + KL in loss
```

Compared with PPO in `04`: **no value model, no GAE.** Compared with RLOO in `03`: the same multiple completions per prompt, with PPO-style clipped updates added.

---

## Timing: GRPO's version of the training step

```
1–2. sample prompts; generate G completions per prompt; SAVE log-probs   ← π_old
3.   score each (reward model or verifier) → R
4.   (if using KL) reference log-probs for the same tokens
5.   per prompt: mean and std of its G rewards → one advantage per completion
     ★ needs all G scored first; copied to every token; FIXED for the batch
     ★ KL is NOT folded into the reward here (unlike PPO)
┌─ inner loop (like PPO; K can be 1) ────────────────────────────────────────
│ 6. current log-probs → per-token ρ
│    loss = −[ clipped(ρ × A) − β·KL ] averaged over tokens, then completions
│    backward
│ 7. optimizer step → ★ policy CHANGES ★
└────────────────────────────────────────────────────────────────────────────
→ new batch
```

*(My addition: many GRPO setups take just **one** gradient step per batch. If that step also uses **no minibatching**, ρ = 1 throughout, clipping never activates, and GRPO behaves like RLOO with z-scored advantages, which shows how close the two are. Clipping only does work once the model has drifted from π_old, through extra epochs **or** minibatches. Lecture 4 `02` covers this in detail.)*

---

## Slides: Conclusion

### "Lecture summary"

The most important sentence on the slide:

> **"In most RLHF setups, data quality and reward signal quality dominate; algorithm choice mostly determines stability, efficiency, and engineering burden."**

In other words, a good reward model or verifier and good prompts matter more than REINFORCE vs. PPO vs. GRPO. The algorithm decides how smoothly and cheaply you get there.

All of them optimize **the same policy gradient** from `01`:

```
∇J(θ) = E[ Σ_t  Ψ_t × ∇ log π_θ(a_t | s_t) ]
```

They differ in exactly three ways:
1. **What Ψ (the advantage) is:** total return, baselined return, critic + GAE, group z-score
2. **How updates are bounded:** none (REINFORCE/RLOO), objective clipping (PPO/GRPO/GSPO), weight capping (CISPO)
3. **Memory and compute:** 2–4 models, with or without training a critic

### "RL cheatsheet"

A one-page reference of all the loss functions: **rlhfbook.com/rl-cheatsheet**

### "Next lecture: RL implementation"

Lecture 4 turns these into code:
- loss implementations for each algorithm
- **loss aggregation**: per-sequence vs. per-token averaging (the 1/|a_i| term above) and why it matters
- infrastructure: what to store (log-probs, values, reference log-probs) and what breaks if you don't
- monitoring training, spotting reward hacking, fixing common failures

Code: Lambert's repo at **rlhfbook.com/code**, which compares the algorithms on toy environments.

---

## The whole family in one table

| | REINFORCE | RLOO | PPO | GRPO | GSPO | CISPO |
|---|---|---|---|---|---|---|
| **Baseline** | simple average | other K−1 | learned critic | group mean ÷ std | group (like GRPO) | group (like GRPO) |
| **Advantage per** | completion | completion | **token** | completion | completion | completion |
| **Ratio** | – | – | per token | per token | **per completion** | per token |
| **Bounding** | none | none | clip (off switch) | clip (off switch) | clip (off switch) | **cap (never off)** |
| **Critic?** | no | no | **yes** | no | no | no |
| **Models** | 3 | 3 | **4** | 2–3 | 2–3 | 2–3 |
| **From** | 1992 | 2024 (RLHF) | 2017 | 2024 (DeepSeek) | 2025 (Qwen) | 2025 (MiniMax) |

---

## The whole section in four lines

1. **GRPO = PPO's clipping + RLOO's group baseline − the critic.** Sample G answers per prompt; advantage = z-score within the group, copied to every token.
2. **It only learns where answers disagree.** If all G answers get the same reward, every advantage is 0. KL moves into the loss and is optional, so it runs with 2–3 models instead of PPO's 4.
3. **GSPO** uses one ratio per *answer* (a geometric mean) instead of per token, so single tokens that swing wildly (especially in MoE models) don't get clipped individually. **CISPO** caps the ratio instead of switching tokens off, so every token always learns.
4. **The lecture's punchline:** every algorithm is the same policy gradient with a different Ψ and a different way of bounding updates. Data and reward quality matter more than which one you pick.
