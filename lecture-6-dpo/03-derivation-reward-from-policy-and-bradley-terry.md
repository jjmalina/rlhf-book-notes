# Derivation Parts 2–3: The Reward from the Policy, and Bradley-Terry

Notes on *Direct Preference Optimization (DPO) and Friends | Post-Training Course, Lecture 6* (Nathan Lambert), 19:19–24:23. Based on `source/lecture-6-dpo-slides.md` and `source/lecture-6-dpo-transcript.md`. Follows on from `02-derivation-the-optimal-policy.md`. **This file merges two short YouTube chapters**: P2/4 "Recovering the reward from the policy" (19:19–21:11, ~2 min) and P3/4 "Connecting to Bradley-Terry preference" (21:11–24:23, ~3 min). They're one idea: get a reward out of the policy, then show the uncomputable Z drops out.

---

## Where this fits

`02` found the best policy **if you knew the reward**. But we don't have a reward. We have **preference pairs**. This chapter bridges the two:

- **Step 2:** turn π* inside out, so the reward is written **in terms of the policy**.
- **Step 3:** plug that into the **Bradley-Terry** model, which is how preference pairs are tied to rewards. The uncomputable Z(x) cancels.

**When does this happen?** Still never at runtime. It's justification for the loss in `04`. But one result here, the **implicit reward**, *is* something you compute and log during training (`05`, `07`).

---

# Part 2: Recovering the reward from the policy (19:19–21:11)

## Slide: "Invert π* for the implicit reward"

The goal: solve the π* formula for r. Four lines of algebra:

| Line | Move |
|---|---|
| π*(y\|x) = (1/Z(x)) · π_ref(y\|x) · exp(r*(x,y)/β) | The answer from `02` (r* = the "true" reward that π* is optimal for) |
| log π* = log( (1/Z) · π_ref · exp(r*/β) ) | Take the log of both sides |
| log π* = −log Z + log π_ref + r*/β | log(abc) = log a + log b + log c. The log and exp cancel on the last term |
| r*/β = log π* − log π_ref + log Z | Move terms across to get r on its own |
| **r*(x, y) = β · log( π*(y\|x) / π_ref(y\|x) ) + β · log Z(x)** | Multiply by β |

The boxed result, which the slide labels **"your LM is secretly a reward model"**:

```
r*(x, y) = β · log( π*(y|x) / π_ref(y|x) )   +   β · log Z(x)
           └──────── the implicit reward ───────┘   └ same for every y ┘
```

**In words:** the reward of a response is **β × (how much more likely the trained policy makes it than the reference did, in log terms)**, plus a constant that is the same for every response to this prompt.

- Policy likes it **more** than the reference did → log-ratio > 0 → **high** reward.
- Policy likes it **less** → log-ratio < 0 → **low** reward.

### Lambert's emphasis: DPO does not train a reward model

"Part of what is forgotten about the DPO paper is that its title is literally *Your Language Model is Secretly a Reward Model*." DPO isn't training a reward model. It's that **any policy tuned from a reference can be read as an implicit reward model**: compare the tuned model's log-probs with the original (SFT) model's. His team used exactly this when building **RewardBench**, to test whether DPO models' implicit rewards work as reward models.

He also notes Bradley-Terry is "just the one that is used most often." The DPO trick could be done with other preference models.

### The toy example, continued *(my addition)*

From `02`, with β = 1: π_ref = (0.5, 0.3, 0.2), rewards (0, 1, 2), π* = (0.179, 0.292, 0.529), Z = 2.793.

| Response | β·log(π*/π_ref) | + β·log Z = 1.027 | = reward |
|---|---|---|---|
| A | −1.027 | +1.027 | **0** ✓ |
| B | −0.027 | +1.027 | **1** ✓ |
| C | +0.973 | +1.027 | **2** ✓ |

The formula recovers the rewards exactly. Now notice: **without** the log Z column, the rewards are (−1.027, −0.027, 0.973). All shifted by the same amount, but **the gaps are identical**: B − A = 1, C − B = 1. If you only ever compare two responses to the same prompt, you don't need Z. That's Part 3.

---

# Part 3: Connecting to Bradley-Terry (21:11–24:23)

## Slide: "Recall the Bradley-Terry model"

Bradley-Terry is the standard model linking rewards to preferences, the same one used to train reward models (lecture 2 / book chapter 5):

```
p*(y₁ ≻ y₂ | x) = exp(r*(x, y₁)) / ( exp(r*(x, y₁)) + exp(r*(x, y₂)) )
```

"y₁ ≻ y₂" means "y₁ is preferred to y₂." In words: **each response gets a score e^reward, and the chance y₁ wins is its share of the two scores.** Equal rewards → 50/50. y₁'s reward much higher → near 100%.

The move: substitute the implicit reward from Part 2 for r*.

## Slide: "Substitute the reward, then cancel Z(x)"

After substituting, every exponent has the form β·log(π*/π_ref) **+ β·log Z(x)**. "It's like, holy crap. Look at this... you never wanna do this by hand, but let's follow it through."

The cancellation uses **e^(a+b) = e^a · e^b**:
- Every term (the numerator and both terms of the denominator) carries the same factor e^(β·log Z) = Z^β.
- Factor it out of top and bottom. It cancels.

```
p*(y₁ ≻ y₂) = e^(Δ₁) / ( e^(Δ₁) + e^(Δ₂) ),    where Δᵢ = β · log( π*(yᵢ|x) / π_ref(yᵢ|x) )
```

"**The intractable term is gone.**" Lambert: "the partition function is there, but it ends up not getting in the way at all."

*(Correction: the transcript has Lambert saying "exponential of a plus exponential of b equals exponential of a times b." That's a slip. The rule used, as the slide says, is e^(a+b) = e^a · e^b: an exponential of a **sum** splits into a **product**, which lets the shared Z^β factor out.)*

*(My note on why it cancels, in plain terms: Z is the same for both responses because they answer the same prompt. Bradley-Terry only cares about the **gap** between two rewards, and a constant added to both doesn't change the gap. That's the toy example above. It's also why DPO pairs must share a prompt.)*

## Slide: "Divide through to a sigmoid"

Multiply top and bottom by e^(−Δ₁):

```
p* = (e^(Δ₁)·e^(−Δ₁)) / (e^(Δ₁)·e^(−Δ₁) + e^(Δ₂)·e^(−Δ₁)) = 1 / (1 + e^(Δ₂ − Δ₁))
```

That's the **sigmoid**, σ(z) = 1/(1 + e^(−z)), with z = Δ₁ − Δ₂:

```
┌────────────────────────────────────────────────────────────────────────────────┐
│  p*(y₁ ≻ y₂ | x) = σ( β·log(π*(y₁)/π_ref(y₁))  −  β·log(π*(y₂)/π_ref(y₂)) )   │
└────────────────────────────────────────────────────────────────────────────────┘
```

**"A sigmoid of the difference of two log-ratios: Bradley-Terry, with the policy in place of the reward model."** y₁ is the chosen response, y₂ the rejected one. "This is the core of what we end up going to implement in the loss function."

### What the sigmoid does *(my addition)*

| Margin z = Δ₁ − Δ₂ | σ(z) = predicted chance y₁ wins |
|---|---|
| −2 | 0.12 |
| 0 | 0.50 |
| +1 | 0.73 |
| +2 | 0.88 |
| +5 | 0.99 |

It's the same function as logistic regression from a first ML course: a score in, a probability out. DPO is, in effect, **logistic regression on "which response wins," where the score is the policy's implicit reward gap.**

---

## The whole section in four lines

1. **Invert π*:** taking logs and rearranging gives **r = β·log(π/π_ref) + β·log Z(x)**. A tuned policy *is* a reward model: "your LM is secretly a reward model."
2. **β·log(π/π_ref)** is the **implicit reward**: how much more the policy likes a response than the reference did. log Z is the same for every response to a prompt.
3. **Plug into Bradley-Terry** (chance y₁ wins = its share of e^reward). Every term carries the same Z^β, so **Z cancels**. Only reward *gaps* matter, and Z shifts both rewards equally.
4. **Result:** p(chosen ≻ rejected) = **σ(implicit reward of chosen − implicit reward of rejected)**. Next (`04`), turn that into a loss.
