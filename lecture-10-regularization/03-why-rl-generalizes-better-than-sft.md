# Why RL Generalizes Better than SFT

Notes on *Regularization in RL, Why RL Generalizes, and Why SFT Forgets | Post-Training Course, Lecture 10* (Nathan Lambert), 21:09–25:15. Based on `source/lecture-10-regularization-slides.md` and `source/lecture-10-regularization-transcript.md`. Follows on from `02-rl-as-a-reverse-kl-loss.md`.

---

## Where this fits

`02` gave the math: SFT is forward KL to fixed data, RL is reverse KL with samples from itself. This chapter gives the **evidence**: three papers showing that RL post-training **forgets less and generalizes better** than SFT, and the explanation for why the naive "mass-covering should forget less" intuition is backwards.

The one sentence to hold onto (from the RL's Razor slide):

> "Among the many high-reward solutions for a new task, on-policy methods such as RL are inherently biased toward solutions that remain closer to the original policy in KL divergence."

This is the **implicit** regularization, and it works **even with β = 0**. That's why lecture 5's reasoning recipes could drop the KL penalty and still not wreck the model.

---

## Part 3 section break: "Why RL generalizes more"

Lambert: "this is very much related to this RL... numerical stability math."

---

## Slide: "SFT memorizes, RL generalizes" (Chu et al., 2025)

A controlled study: post-train on one task, then test on a **rule shift**, a small change to the problem that makes it out-of-distribution (OOD).

| Task | Train on | Test with the shift |
|---|---|---|
| **GeneralPoints** | Reach 24 by combining four cards with +, −, ×, ÷. Face cards J/Q/K = 10 | J/Q/K = 11/12/13 |
| **V-IRL** | Visual navigation with **absolute** directions (north/east) | **Relative** directions (left/right) |

The headline numbers on V-IRL, out-of-distribution accuracy:

| Method | Before | After |
|---|---|---|
| **RL** | 80.8% | **91.8%** (improves) |
| **SFT** | 80.8% | **1.3%** (collapses) |

SFT didn't just fail to generalize. It **destroyed** "spatial reasoning the base model already had." It learned the training rule so literally that any change broke it: that's what "memorizes" means here.

Lambert: "RL on these simple problems generalizes very well out of distribution... whereas you SFT on a narrow distribution, the generalization capability crashes." He's careful to call it "a kind of a toy problem," but a clean one where you can "understand the background math and test it."

The slide's conclusion: **"RL-based post-training carries *implicit* regularization from its on-policy structure alone. SFT is more brittle."**

---

## Slide: "Which direction should forget less?" (the question)

The naive reading of `02`:

> Forward KL is *mass-covering*, so SFT should preserve every mode. Mode-seeking RL should collapse onto one and forget the rest.

"Is this correct?" Lambert: "this intuition is backwards, and I think partially this is just because the words being used are pretty bad."

---

## Slide: "Which direction should forget less?" (the answer, Chen et al., "Retaining by Doing", 2025)

The slide: "That intuition assumes a **unimodal** policy. LLMs are **multimodal**."

*(Figure reconstructed from the slide caption and transcript.)* The setup: the model π_θ already has an **old mode** (what it knows, which we want to keep). The target π* has that same old mode **plus a new target mode** somewhere else (the new task). The model also has a small secondary mode that's in the wrong place.

```
START          π_θ:   ███ old mode           ▂ small mode (in the wrong spot)
               π*:    ███ old mode                          ███ new target mode

FORWARD KL     π_θ:   ██▂ old mode, shrunk   ▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂  stretched wide to cover target
(SFT)                 ↑ mass stolen from here to blanket the gap
                      "the new mode stretches out to be really wide and it can even take mass
                       in the other mode"

REVERSE KL     π_θ:   ███ old mode, untouched                ███ small mode MOVED onto target
(RL)                  "mode seeking... can move more nicely and get there"
```

Lambert's reading:
- **Forward KL (SFT)** "flattens the secondary mode, because it's trying to make sure that it doesn't have any zero weight in the wrong area." Covering everywhere the target has mass is expensive, and the cheapest way to pay for it is to **borrow mass from the old mode**. That's forgetting.
- **Reverse KL (RL)** "can truly send probabilities to zero and ignore things." It can **slide one mode over** to the new target and leave the old mode exactly as it was.
- "**Mass covering is kind of blanketing and mode seeking is moving over.**"
- He admits "this diagram is not perfectly intuitive, but it gets the point across."

### Why the naive intuition fails (my explanation)

The textbook picture (a one-bump model fitting a two-bump target) is where "reverse KL drops a mode" comes from. With one bump, reverse KL has to sit on one peak and **abandons the other**: that would be forgetting. But an LLM is not one bump. It can represent many modes at once, and then:

| | What "covering" actually covers | Effect on old knowledge |
|---|---|---|
| **Forward KL (SFT)** | the **new dataset**, every token of it | The loss contains **no term about the old behavior at all**. Old modes are only kept by luck, and they're the first place to borrow mass from |
| **Reverse KL (RL)** | only where the model's **own samples** go | Samples come from the current model, mostly its existing modes. Modes it doesn't need to change aren't pushed on, so they stay |

"Mass-covering" sounds protective, but it's protective of the **wrong distribution**: the new data, not the old model.

---

## Slide: "RL's razor: Lower KL drift for equivalent performance" (Shenfeld et al.)

> *"Among the many high-reward solutions for a new task, on-policy methods such as RL are inherently biased toward solutions that remain closer to the original policy in KL divergence."*

Two findings:

1. **Forgetting tracks KL drift.** Forgetting ≈ f( E_{x ~ new task}[ D_KL(π_0 ‖ π) ] ), with **R² = 0.96**. In words: measure how far the fine-tuned model π has moved from the original π_0, **on the new task's prompts only**, and that one number predicts how much old-task performance was lost. "A cheap forgetting predictor": you don't even need to run the old benchmarks.
2. **On-policy data is the whole story.** The ablation: "on-policy data fully accounts for the difference. Negative gradients have no discernible effect." It's not that RL can push bad answers down (SFT can't). It's that RL **trains on its own samples**.

*(Figure reconstructed from the caption.)* Of all the policies that solve the new task, RL converges to the ones **closest in KL to the base model**, which keeps more prior-task ability at the same new-task score.

```
                       all policies that solve the new task
                  ┌──────────────────────────────────────────┐
   base π_0 ●─────┼──► ● RL lands here (small KL)             │
                  │                                           │
                  │                       ● SFT lands here    │
                  │                         (large KL)        │
                  └──────────────────────────────────────────┘
```

Lambert: "if you have two models, one trained with RL and one trained with SFT, where they end up having similar performance... the RL can get to the policy with similar performance while moving less through the KL divergence landscape." "The RL policies are biased towards ones that are a bit simpler." He's mildly cautious: "the results are probably okay."

*(Corrections/notes: the transcript's "Retraining by doing" is the paper "**Retaining** by Doing: The Role of On-Policy Data in Mitigating Forgetting." The slide cites RL's Razor as Shenfeld et al., 2026. Its arXiv ID, 2509.04259, means it was first posted in September 2025; 2026 is presumably the published version. Also note the direction of the predictor's KL: D(π_0 ‖ π), with the **original** model in the first slot, measured on new-task inputs.)*

---

## Why SFT forgets and RL stays close: the mechanism, step by step (my addition)

Put `02` and this chapter together, in terms of what the loss actually pushes on each step.

**SFT** pushes up −log π_θ(**data's** token). The data was written by someone else (a human, a teacher model).
- If the model finds a data token unlikely, say probability 0.01, its loss is −log 0.01 ≈ **4.6**: a large push. Data tokens in a style the model would never use get the **biggest** pushes.
- The data never changes, so the pushes keep pointing at the same far-away target, step after step, until the model imitates it. It moves as far as needed, wherever that is.

**RL** pushes up advantage × log π_θ(**own** token).
- Every token it trains on was **sampled by the model**, so it's something the model already finds plausible (typical probabilities, not 0.01-surprises).
- Reinforcing what you already sometimes do is a **small move**. And after each update, the next batch is sampled from the updated model, so the "target" moves with it rather than being fixed somewhere far away.

### A 3-answer illustration of RL's Razor (my addition)

One new-task prompt. The original model π_0 gives three answers. **B and C are both correct** (reward 1). A is wrong.

| Answer | π_0 | Correct? |
|---|---|---|
| A | 0.50 | ✗ |
| B | 0.30 | ✓ |
| C | 0.20 | ✓ |

- **SFT** on a teacher's dataset that always writes **C** (a correct answer, but in a style the model rarely uses) ends up near {A 0.05, B 0.05, C 0.90}.
- **RL** samples from π_0 itself. B shows up 30% of the time, C 20%, and both get rewarded, so mass shifts from A to B and C **roughly in the proportions the model already had**: about {A 0.05, B 0.57, C 0.38}.

| | P(correct) | D(π_0 ‖ π), RL's Razor's measure |
|---|---|---|
| SFT result | 0.95 | **1.39** |
| RL result | 0.95 | **0.83** |

Same task score, but RL moved **much less** (and the reverse direction, D(π ‖ π_0), shows the same gap: 1.15 vs. 0.50). RL's Razor says that smaller number predicts less forgetting. Note RL's result is just π_0 with the wrong answer pushed down and the correct ones scaled up together. That's the reward-tilted reference from `02` (with a small β), reached here by sampling.

### Connection to β = 0 in lecture 5

Lecture 5's trick #4: short single-turn RLVR runs often set **β = 0**, with no reference model at all (`../lecture-5-reasoning-models/03-common-implementation-patterns.md`). This chapter is why that works:

| What keeps the model near home | Present at β > 0? | Present at β = 0? |
|---|---|---|
| Explicit KL penalty (`01`) | ✅ | ❌ |
| Exact "RL = reverse KL to π*" identity (`02`) | ✅ | ❌ ("this theory does not apply") |
| **On-policy sampling bias (RL's Razor)** | ✅ | ✅ |
| Hard-to-hack verifiable reward (lecture 5 `01`) | – | ✅ in RLVR |

So β = 0 removes the leash, but RL still mostly walks. The KL in the β = 0 OLMo run in `01` rose and then wandered rather than exploding: that's the implicit regularization showing up in a real log.

---

## When does this happen? (timing view)

The implicit regularizer lives in **step 2** (who writes the training data). The forgetting measurement lives **after training**:

```
TRAINING (each batch)
  2. generate with current π_θ   ← RL: the data comes from HERE (implicit regularizer)
                                   SFT: skips this; data comes from a fixed file
  6. loss on those tokens
  7. update → next batch sampled from the updated model

AFTER TRAINING (evaluation)
  run π and π_0 on the NEW task's prompts → D(π_0 ‖ π)      ← RL's Razor: predicts forgetting (R² 0.96)
  run on shifted / old tasks → OOD accuracy, retention     ← Chu et al.: RL 80.8→91.8, SFT 80.8→1.3
```

---

## The whole section in five lines

1. **"SFT memorizes, RL generalizes"** (Chu et al.): on a rule-shifted navigation task, RL went 80.8% → 91.8% OOD, SFT collapsed to 1.3%.
2. **The naive intuition is backwards.** "Mass-covering" SFT covers the **new data**, not the old model, and steals mass from old modes to do it. Mode-seeking RL can slide one mode to the new target and leave the rest untouched ("blanketing vs. moving over").
3. **The textbook "reverse KL drops modes" picture assumes a one-bump model.** LLMs are multimodal.
4. **RL's Razor:** among equally good solutions, RL finds the one closest in KL to the base model. Forgetting tracks D(π_0 ‖ π) on new-task data (R² = 0.96), and on-policy data alone explains the difference.
5. **That's why β = 0 is survivable** in short RLVR runs: the explicit leash is gone, but on-policy sampling still keeps the model near home.
