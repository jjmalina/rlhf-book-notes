# Derivation Part 4: The Loss and the Gradient

Notes on *Direct Preference Optimization (DPO) and Friends | Post-Training Course, Lecture 6* (Nathan Lambert), 24:23–30:16. Based on `source/lecture-6-dpo-slides.md` and `source/lecture-6-dpo-transcript.md`. Follows on from `03-derivation-reward-from-policy-and-bradley-terry.md`. Includes a **worked numeric example** of the DPO loss and a comparison with lecture 3's "weighted supervised learning."

---

## Where this fits

`03` ended with a formula for **the probability that the chosen response beats the rejected one**, written using only the policy and the reference. This chapter does the last two things:

1. **The loss:** make the preferences we actually observed as likely as possible.
2. **The gradient:** see what that loss does to the weights, and recognise it as lecture 3's weighted supervised learning.

**When does this happen?** This is the first part that actually runs. The loss is computed at **step 6** of every training step (`00-map.md`), and `loss.backward()` computes the gradient. As in lecture 3, you never write the gradient yourself: "what you actually implement is loss and then the automatic differentiation packages of PyTorch... implement said gradient when you call backwards." The gradient slides are for **understanding** what the loss does.

---

## Slide: "The DPO loss"

**Step 1: negative log-likelihood of the preferences.**

```
L_DPO = −E_{(x, y_c, y_r) ~ D}[ log p(y_c ≻ y_r | x) ]
```

In words: "we want to increase the probability of this. We want to make this likely." The data says y_c beat y_r. A good model should give that event a high probability. Take the log (easier to work with), and put a minus sign in front because a loss is something you minimize. It's the same as the **log-loss** for logistic regression, and the same recipe used to train a Bradley-Terry reward model.

**Step 2: plug in the sigmoid from `03`, with the trainable policy π_θ in place of π*.**

```
L_DPO = −E[ log σ( β·log(π_θ(y_c|x)/π_ref(y_c|x))  −  β·log(π_θ(y_r|x)/π_ref(y_r|x)) ) ]
                   └──── implicit reward of chosen ───┘     └─── implicit reward of rejected ──┘
```

*(My note on the swap from π* to π_θ: `03` said "if a policy is optimal for some reward, its log-ratio is that reward." DPO turns that around: treat the policy you're training **as if** it were optimal, read off its implicit reward, and adjust the policy until those implicit rewards explain the preference data well. Fitting the policy *is* fitting the reward.)*

"This is directly differentiable — **no reward model, no sampling, no RL loop.**" Lambert: "it's kind of a mess... but it is differentiable."

### Reading the loss piece by piece

| Piece | What it is | Where it comes from at runtime |
|---|---|---|
| log π_θ(y_c\|x), log π_θ(y_r\|x) | The policy's log-prob of the **whole** chosen / rejected response: the sum of its token log-probs | Step 6: a forward pass of the policy over the given text (like SFT) |
| log π_ref(y_c\|x), log π_ref(y_r\|x) | The same, for the frozen reference | **Cached** before training, or one forward pass (no gradients) |
| β·(policy − reference) | The **implicit reward**: how much more the policy likes it than the reference did | Arithmetic |
| chosen reward − rejected reward | The **margin** u | Arithmetic |
| −log σ(u) | Small when the margin is large and positive; large when it's negative | Arithmetic → the loss |

---

## A worked numeric example *(my addition)*

**Setup:** one prompt, one pair. **β = 0.1** (a common DPO value). Log-probs are for whole responses, so they're big negative numbers (a 50-token response where each token has probability ~0.37 has log-prob ~−50).

**Reference** (frozen, cached once): log π_ref(y_c) = **−50**, log π_ref(y_r) = **−48**. The reference slightly *prefers the rejected answer*. That's realistic: the reference is the SFT model, before it has learned this preference.

The formulas, applied to each scenario:

```
chosen reward    r_c = β × (log π_θ(y_c) − log π_ref(y_c))
rejected reward  r_r = β × (log π_θ(y_r) − log π_ref(y_r))
margin           u   = r_c − r_r
loss                 = −log σ(u)
gradient weight  w   = σ(−u)   = σ(r_r − r_c)    ("how wrong", explained below)
```

| Scenario | log π_θ(y_c) | log π_θ(y_r) | r_c | r_r | margin u | σ(u) | **loss** | **w** |
|---|---|---|---|---|---|---|---|---|
| **A. Start of training** (π_θ = π_ref) | −50 | −48 | 0.0 | 0.0 | 0.0 | 0.50 | **0.693** | 0.50 |
| **B. Learned it** | −45 | −55 | +0.5 | −0.7 | +1.2 | 0.77 | **0.263** | 0.23 |
| **C. Got it backwards** | −52 | −46 | −0.2 | +0.2 | −0.4 | 0.40 | **0.913** | 0.60 |

Working scenario B by hand:
- Chosen: −45 − (−50) = +5, × 0.1 = **+0.5**. The policy likes the chosen answer more than the reference did.
- Rejected: −55 − (−48) = −7, × 0.1 = **−0.7**. The policy likes the rejected answer less.
- Margin: 0.5 − (−0.7) = **1.2**. σ(1.2) = 1/(1 + e^(−1.2)) = 1/(1 + 0.301) = **0.768**. Loss = −ln(0.768) = **0.263**.

**The same thing via the code** (`01`'s four lines), to show they agree:

```
pi_logratios  = −45 − (−55) = 10
ref_logratios = −50 − (−48) = −2
logits        = 10 − (−2)  = 12
loss          = −logsigmoid(0.1 × 12) = −log σ(1.2) = 0.263
```

The code groups things differently (chosen − rejected first, then policy − reference), but it's the same arithmetic.

What to notice:
- **At the start, every pair's loss is log 2 = 0.693** and the margin is 0, whatever the data. The policy and reference are identical, so every implicit reward is 0. That's a useful sanity check on a real run.
- **The reference's own preference is "subtracted out."** In scenario B the reference preferred rejected by 2 nats, so the policy had to swing 12 nats (from −2 to +10) to get the margin of 1.2. DPO rewards *change relative to the reference*, not raw likelihood.
- **β converts log-ratios into rewards.** With β = 0.1, a 10-nat swing is only a 1.0 margin. A larger β makes the same swing count for more, so the loss is satisfied with **less movement from π_ref**. That's how β keeps the KL leash (`05`).

(One more scenario, where the loss improves **even though the chosen response got less likely**, is in `05`: likelihood displacement.)

---

## Slides: "The DPO gradient" (the three slides)

The slides do the calculus. Here's what each step means, in words.

| Step on the slide | Plain meaning |
|---|---|
| Let u = the margin (the inside of σ) | Name the thing inside, to keep it readable |
| Chain rule: d/dz log σ(z) = σ′(z)/σ(z) | "How the loss changes" = (how log σ changes with the margin) × (how the margin changes with the weights) |
| σ′(u) = σ(u)(1 − σ(u)), so the ratio is 1 − σ(u) | A standard sigmoid identity. "These things are everywhere" |
| 1 − σ(u) = σ(−u) | Another identity: the sigmoid is symmetric |
| ∇u = β·[∇log π_θ(y_c) − ∇log π_θ(y_r)] | **π_ref is frozen**, so its terms have no gradient and drop out. Only the policy's log-probs are left |

The result:

```
∇L_DPO = −β · E[  w  ·  ( ∇log π_θ(y_c|x) − ∇log π_θ(y_r|x) )  ]
                  │        └───────── "which knobs" for chosen, minus "which knobs" for rejected
                  └ w = σ( r_θ(x, y_r) − r_θ(x, y_c) ),   with r_θ = β·log(π_θ/π_ref)
```

Gradient descent moves the weights **against** ∇L. The leading minus cancels out, so each step moves the weights by **+β·w·(∇log π(chosen) − ∇log π(rejected))**: up on chosen, down on rejected.

### The three pieces (the slide's bullets)

| Piece | Meaning |
|---|---|
| **w ∈ (0, 1)**, "how wrong" | **Larger when the model is more wrong**: when its implicit reward ranks the rejected response above the chosen one. In the example: 0.60 when backwards (C), 0.50 at the start (A), 0.23 once it's learned (B) |
| **The bracket** | **Raise the likelihood of y_c and lower that of y_r.** Lambert: it "raises the likelihood of the tokens rather than the rewards" |
| **β** | Scales the step, and trades getting the order right against drift from π_ref (inherited from the KL penalty) |

*(Clarification: in the transcript Lambert says "the more different the yRejected and the yChosen are, the bigger the weight should be." The direction matters: w is big when the **rejected** response has the higher implicit reward (the model is wrong), and **small** when the chosen one is well ahead, even though they're "different" then too. The slide's wording, "larger when the model is more wrong," is the accurate one.)*

*(Lambert wonders aloud whether the two βs in the gradient are an error. They aren't, as far as I can tell. One β is **inside** w, because the implicit rewards are β·log-ratios. The other is **outside**, from differentiating the β in u. The original DPO paper's gradient has the same two βs.)*

*(My note: w never reaches exactly 0. Even a pair the model already gets right keeps pushing the margin wider, just more gently. With a small, clean dataset the margins can keep growing, which is one motivation for IPO in `06`.)*

---

## The DPO gradient as weighted supervised learning *(my addition)*

Lecture 3's central idea (`../lecture-3-policy-gradients/01-policy-gradients-core-intuitions-and-maths.md`, in code in `../lecture-4-implementing-rl/01-translating-the-policy-gradient-into-code.md`):

> **RL loss = −advantage × log π(token)**. Each response's "which knobs" vector is weighted by a number. SFT is the special case where the weight is always 1.

The DPO gradient has **exactly this shape**, with two responses per prompt:

| Response | Its "which knobs" vector | Weight on it (acts like an advantage) |
|---|---|---|
| chosen y_c | ∇log π_θ(y_c) | **+β·w** (push up) |
| rejected y_r | ∇log π_θ(y_r) | **−β·w** (push down) |

So DPO is equivalent to this hand-written loss (for the gradient only; `w` treated as a fixed number):

```
loss ≈ −β·w·log π_θ(y_c)  +  β·w·log π_θ(y_r)
```

| | SFT | RL (REINFORCE / GRPO) | **DPO** |
|---|---|---|---|
| Whose text | human / teacher | the **model's own samples** | **given pairs** (anyone's) |
| Weight ("advantage") | always +1 | reward − baseline, from an RM or verifier | **+βw on chosen, −βw on rejected** |
| Where the weight comes from | nothing | a **reward model / verifier**, at step 3 | **the model's own current mistake** (implicit reward gap vs. π_ref) |
| Can push probability down? | no | yes, when advantage < 0 | **yes, always, on the rejected response** |
| Weight changes during training? | no | each batch (new samples) | **yes: shrinks as the model gets each pair right** |

*(My note: a close RL analogue is RLOO with K = 2 (`../lecture-3-policy-gradients/03-reinforce-and-rloo.md`). With two samples, each one's baseline is the other's reward, so the advantages are +(R_c − R_r) and −(R_c − R_r): equal and opposite, like DPO's ±βw. The differences: in DPO the two texts are fixed data, not fresh samples, and the weight is a sigmoid of the model's own implicit reward gap, not a reward model's score.)*

This is also why the SFT intuition carries over: a DPO step is "SFT toward the chosen answer, plus *anti*-SFT away from the rejected one, both turned up when the model is getting the pair wrong."

---

## Slide: "Recap: The DPO derivation"

```
Objective        :  max E[r(x,y)] − β·KL(π ‖ π_ref)
⇒ Optimal policy :  π*(y|x) = (1/Z(x)) · π_ref(y|x) · exp(r(x,y)/β)                (02)
⇒ Implicit reward:  r*(x,y) = β·log(π*(y|x)/π_ref(y|x)) + β·log Z(x)               (03)
⇒ Bradley-Terry  :  p*(y₁ ≻ y₂) = σ(r*(x,y₁) − r*(x,y₂))        (Z cancels)         (03)
⇒ DPO loss       :  −log σ( β·log(π_θ(y_c)/π_ref(y_c)) − β·log(π_θ(y_r)/π_ref(y_r)) )  (04)
```

"The reward model never had to be built — it was hiding inside the policy the whole time." Lambert's summary of the implementation: "it's kind of just reading off the loss... this log ratio for chosen and log ratio for rejected. But all the others are kind of steps to get this loss function."

---

## The whole section in four lines

1. **The loss is the negative log-likelihood of the observed preferences**: −log σ(implicit reward of chosen − implicit reward of rejected). It's logistic regression on "which one wins."
2. **Worked example (β = 0.1):** at the start every pair's loss is 0.693 (margin 0). Raising chosen by 5 nats and lowering rejected by 7 nats gives margin 1.2 and loss 0.263. Getting it backwards gives loss 0.913.
3. **The gradient = −β · w · (∇log π(chosen) − ∇log π(rejected))**, where w = σ(rejected reward − chosen reward) is "how wrong" the model is: up on chosen, down on rejected, harder when wrong.
4. **That's lecture 3's weighted supervised learning** with weights ±βw. The weight comes from the model's own mistake relative to π_ref, not from a reward model, and the text is fixed data, not the model's own samples.
