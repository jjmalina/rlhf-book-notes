# Part 1: Over-optimization & Goodhart's Law

Notes on *Over-Optimization and RLHF's Bad Reputation | Post-Training Course, Lecture 9* (Nathan Lambert), 04:20–09:57. Based on `source/lecture-9-over-optimization-slides.md` and `source/lecture-9-over-optimization-transcript.md`. Follows on from `01-intro-and-why-we-care.md`.

---

## Where this fits

This is the technical core of the lecture: **what over-optimization is, how it differs from overfitting, and the two classic experiments that measured it.** The slides' plots aren't in the folder, so the charts below are reconstructed from the captions and the transcript.

---

## Slide: "Goodhart's law: The reward is a proxy"

> "Any observed statistical regularity will tend to collapse once pressure is placed upon it for control purposes." (Goodhart, 1984)

Colloquially: **"When a measure becomes a target, it ceases to be a good measure."**

*(My note: Goodhart was an economist writing about monetary policy. The slide cites a 1984 book; the original paper is from 1975 and was reprinted there. The short "measure becomes a target" phrasing came later, from other authors. The slide cites Hoskin, 1996.)*

Lambert thinks the law is **"overblown"** as stated: "it's still useful to optimize for human preferences, but Goodhart's law is what people think about when they think about taking it too far." Optimizing the proxy *does* help, up to a point.

The three bullets:
1. **RL is a very strong optimizer.** It "pulls *all* the available reward out of the environment."
2. **In RLHF the reward is a learned model**, at best *correlated* with downstream quality. "It'll always have imperfections that can be captured by RL."
3. **The shape:** optimizing the proxy makes the true objective **better early in training, then worse.**

And the practical definition:

> "Overoptimization is **the art of stopping your training** when the real signal is going down even though your proxy signals may still look like they're going up."

### An analogy (my addition)

A teacher can't read every essay carefully, so she grades partly on **length**. At first, this works: longer essays usually *do* have more ideas. Then students learn the rule. Essays get longer by repeating themselves. Every grade goes up. The essays get worse. Length stopped being a good measure the moment it became the target.

The RM is that teacher, but with blind spots that are much harder to guess than "length."

---

## Slide: "Over-optimization v. overfitting"

These sound alike but are different problems.

| | **Overfitting** | **Over-optimization** |
|---|---|---|
| What happens | The model **memorizes training examples** instead of learning general patterns | The model **genuinely gets better at the proxy**, but the proxy has drifted away from the true goal |
| Train metric | ↑ | ↑ (RM score on training prompts) |
| Validation metric | **↓** (held-out accuracy falls) | **↑ too** (RM score on held-out prompts *also* rises) |
| What falls | The same metric, on new data | **A different thing**: real user satisfaction |
| Kind of problem | Generalization | **"A measurement/metric problem"** |
| Can a validation split catch it? | Yes | **No**, because the validation set is scored by the same flawed judge |

Lambert: "Even if you had a train and validation set for your proxy objective... overoptimization is still gonna be improving the validation loss of this... It's more of an imprecise, imperfect measurement tool problem."

**Worked contrast (my addition):**
- *Overfitting:* a classifier gets 99% on training images and 70% on new images. Same task, same grader, different data.
- *Over-optimization:* the RM gives the policy 8.1/10 on training prompts **and** 8.0/10 on brand-new prompts. Both went up from 6/10. But when humans compare the new outputs to the old, they now prefer the old ones. The grader was wrong, not the data split.

**What gaming looks like** (slide): "verbose, confident-sounding answers that score well without being more helpful; repeating rare tokens that hit artifacts in RM training." Lambert lists "verbosity, sycophancy, confidence, and other artifacts of just numerical issues."

*(Correction: in the transcript Lambert says overfitting makes a model "more biased," then asks to be corrected. In the bias–variance trade-off, overfitting is **high variance**: the model fits the noise in its particular training set, so it would change a lot with a different training set. High **bias** is the opposite problem, underfitting: a model too simple to fit the pattern.)*

---

## Slide: "The shape of over-optimization"

*(Chart reconstructed from the slide caption and transcript.)*

"The recurring shape of RLHF training runs: the run looks healthy (training reward keeps climbing) but downstream evaluations peak and then decline."

### Sketch: proxy reward vs. true quality as optimization (KL) increases

```
 score
   ▲
   │                                              ..... proxy reward (RM score)
   │                                       .......        keeps climbing
   │                               ........
   │                        .......
   │                  ......
   │             .....      ___----‾‾‾‾‾‾‾----___
   │         ....     __--‾‾                    ‾‾--__        true quality
   │      ...    __-‾‾        ▲                       ‾‾--__  (gold RM / humans /
   │    ..   _-‾‾              │ best checkpoint              ‾‾-_  downstream evals)
   │  .. _-‾‾                  │ STOP HERE                        ‾-_
   │ ._-‾                      │                                     ‾-
   │.-‾                        │
   └───────────────────────────┼──────────────────────────────────────────►
   0              ◄── helps ──►│◄──── hurts: gains come from RM blind spots ────►
                     KL distance from the starting policy (√KL on many plots)
                     (≈ how far training has pushed the model ≈ training steps)
```

Read it left to right:
1. **Early:** the policy fixes real problems the RM correctly rewards. Proxy and true quality rise together.
2. **The peak:** the easy, real improvements are used up.
3. **Late:** the cheapest remaining reward is in the RM's blind spots ("regions of the reward model that do not map to real usage"). The proxy keeps rising; true quality falls.

### A toy run (my addition, made-up numbers)

| Step | KL from start | RM score (proxy) | Human win-rate vs. the SFT model (true) | What the samples look like |
|---|---|---|---|---|
| 0 | 0 | 0.0 | 50% | the SFT model |
| 100 | 1 | 0.8 | 58% | clearer, better structured |
| 200 | 3 | 1.3 | **62%** ← peak | best checkpoint |
| 400 | 8 | 1.9 | 57% | longer, more headers, "Great question!" |
| 800 | 20 | 2.6 | 41% | rambling, flattering, repetitive |

The training dashboard only shows the first three columns. **Every one of them looks like steady progress.** Only the fourth column (which you have to measure separately) shows the problem.

### Why the RM is wrong "out there" (my addition, the timing view)

The RM was trained **in an earlier stage** on comparisons of outputs from roughly the SFT model, then **frozen** (lecture 3 `00` "the cast": the reward model never changes during RL). So:

```
RM training stage:   RM sees text like the SFT model writes → learns to score that kind of text well
RL, step 0:          policy = SFT model → its outputs look like the RM's training data → scores are trustworthy
RL, step 800:        policy has drifted (KL = 20) → writes text the RM never saw → scores are guesses
```

**KL from the reference is roughly "how far outside the RM's experience are we now?"** That's why KL is the x-axis on every one of these plots, and why the KL penalty (lecture 3 `02`) works as a leash: it keeps the policy in the region where the RM is still trustworthy.

**Connects to:** `../lecture-3-policy-gradients/02-wiring-rlhf-for-language-models.md` (the reference model and "reward hacking"). Lecture 3 said the policy "will find and exploit those blind spots." This chart is what that looks like over a run.

---

## Slide: "Scaling laws for RM over-optimization (seminal paper)", part 1

*Scaling Laws for Reward Model Overoptimization* (Gao, Schulman, Hilton). Lambert: "the canonical over-optimization paper," posted in **fall 2022, before ChatGPT**. *(The slide cites it as 2023, its ICML publication year. The arXiv version is October 2022.)*

Two plots side by side *(reconstructed from the captions)*:

| | Left: **Best-of-N** | Right: **RL** (PPO) |
|---|---|---|
| How it optimizes | Sample N answers, keep the one the proxy RM likes best. **No weights change** | Train the policy weights against the proxy RM |
| Caption | "gold reward (solid) flattens and falls while the proxy (dashed) keeps climbing" | "same shape, ~10x more KL spent, and a much harder turnover for small RMs" |

### Best-of-N in plain terms

Best-of-N is the gentlest optimizer: it only **chooses** among answers the unchanged model would write anyway. Lambert: "RL is the only one that is changing the policy parameters."

*(My addition: BoN has a known KL cost, **KL ≈ log N − (N−1)/N** (in nats). It grows only with the log of N:)*

| N | KL spent |
|---|---|
| 4 | 1.39 − 0.75 ≈ **0.64** |
| 16 | 2.77 − 0.94 ≈ **1.83** |
| 64 | 4.16 − 0.98 ≈ **3.17** |

*(Going from 16 to 64 samples, 4× the compute, buys only ~1.3 more nats of optimization. RL runs can go much further, which is part of why "best-of-N sampling spends far less KL than online RL" (`03`).)*

---

## Slide: "Scaling laws for RM over-optimization (seminal paper)", part 2: the setup

**The problem:** to draw the "true quality" curve you need to ask humans about every checkpoint, which is far too expensive. So they **replaced the humans with a bigger model**:

```
6B "GOLD" RM  ──labels the comparison data──►  smaller PROXY RMs (3M – 3B params) trained on those labels
     │                                                    │
     │                                          RL / BoN optimizes the PROXY only
     │                                                    │
     └──────── score the policy with BOTH ◄───────────────┘
               proxy (dashed): climbs forever
               gold  (solid):  peaks and falls
```

The gold RM plays "the humans," and the proxy RM plays "our reward model." Lambert: "they're training a reward model to predict the preferences of another reward model... This is an added layer between reality... But this is the only way to actually establish scaling laws for it."

**Findings on the slide:**
- **Larger proxy RMs turn over later and more gently.** A better RM has fewer exploitable blind spots near the start, so you can push further before it breaks.
- "Since gold-labels-train-proxy is another proxy, **real models diverge slightly differently.**" The exact numbers don't transfer to real human preferences; the shape does.

*(My addition: the paper fits simple formulas to the gold curves, using d = √KL as "distance travelled." For BoN, gold ≈ d × (α − β·d): it rises in a straight line at first, and then the −β·d² part wins, so it's an upside-down parabola. For RL, gold ≈ d × (α − β·log d): the same rise, but the penalty grows more slowly, so the decline is gentler and stretched over more KL. "Scaling laws" means α and β change smoothly and predictably with the proxy RM's size.)*

---

## Slide: "Measuring it: Train vs. test reward models"

*(Chart reconstructed from the slide text and transcript.)*

Anthropic's version, from their 2022 "Helpful and Harmless" assistant paper (Bai et al.):

1. **Split the preference data in half** and train two **52B** preference models: a *train PM* and a *test PM*.
2. **RL against the train PM only.**
3. Score the policy with **both**.
4. The x-axis is **√KL from the initial policy** ("how much the policy has changed").

*(Correction: the transcript says "50,000,000,000 parameter"; the slide and the paper say 52B.)*

What you see: both scores rise at first. Then the **train PM keeps rising while the test PM falls behind**. "As you're training the model more, you're getting gains that are more specific to that reward model rather than the one on the held-out set."

This version is **more realistic** than Gao's: both PMs are trained on real human labels. The test PM doesn't share the train PM's particular blind spots (it saw different data), so the gap between them measures **how much of the gain is exploiting one model's quirks**.

*(My note: Bai et al. reported that the train reward grows roughly **in a straight line with √KL** over much of training. That's why √KL is a natural x-axis: it turns the curve into a straight line, so departures from it are easy to see.)*

**Practical version (my addition):** you can run this check yourself at **step 3**. Score each batch's rollouts with a second RM that the policy isn't trained on, and log both. It costs an extra forward pass through a second RM per batch, but it gives you a "true quality" curve inside the training loop instead of only at checkpoint evals.

Lambert's aside on the field: "I haven't seen as much awesome work on this since then... the art of training reward models never really moved from industry to academia." Even with benchmarks like RewardBench (which he worked on), "I don't think we ever got great at training these reward models" in academia.

---

## The whole section in five lines

1. **Goodhart's law:** "When a measure becomes a target, it ceases to be a good measure." RL is strong enough to pull every bit of reward out of a learned RM, including the reward from its flaws.
2. **Over-optimization ≠ overfitting.** The proxy goes up on validation data too. What falls is a *different* quantity, real quality. It's a measurement problem, so a validation split can't catch it.
3. **The shape:** plotted against KL from the start, the proxy rises forever; true quality rises, peaks, and falls. The job is to stop near the peak.
4. **Gao et al.:** a 6B gold RM stands in for humans. BoN and RL show the same shape; RL spends ~10× more KL, and bigger proxy RMs break later and more gently.
5. **Anthropic:** split real preference data, train two 52B PMs, RL on one, watch the other fall behind. You can copy this at step 3 with a held-out RM.
