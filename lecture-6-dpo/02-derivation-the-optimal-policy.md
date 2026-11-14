# Derivation Part 1: The Optimal Policy

Notes on *Direct Preference Optimization (DPO) and Friends | Post-Training Course, Lecture 6* (Nathan Lambert), 08:01–19:19. Based on `source/lecture-6-dpo-slides.md` and `source/lecture-6-dpo-transcript.md`. Follows on from `01-why-dpo-is-a-big-deal.md`. See `00-map.md` for the derivation in five plain-language steps.

---

## Where this fits

This is the longest and hardest chapter: **step 1 of the derivation.** The goal is to find, *on paper*, the policy that best solves the RLHF objective. No calculus is needed, only log rules and one fact about KL divergence. Lambert: "it is a lot of algebra, but it's pretty straightforward."

**When does this happen?** Never at runtime. Like lecture 3's log-derivative trick, this derivation only **justifies** the loss in `04`. Nothing in this file is computed during training. (In particular, π* and Z are never computed.)

---

## The plan for this chapter, before any symbols

1. Write down the RLHF goal: high reward, small drift from π_ref.
2. Rearrange it (pure algebra) until it looks like **"how far is π from some target distribution q?"**
3. Use the fact that **distance from q is smallest (zero) when π = q.**
4. So the answer is π* = q. Read off what q is.

The trick is all in step 2: rewriting the objective so the answer becomes obvious.

### Three facts used everywhere below *(my addition)*

| Fact | Plain meaning |
|---|---|
| log(a/b) = log a − log b, and log(ab) = log a + log b | Logs turn division into subtraction and multiplication into addition |
| c = log(e^c) | Any number can be written as "the log of e to that number." Used to sneak a term inside a log |
| **KL(π ‖ q) = average over y ~ π of log(π(y)/q(y))**. It is ≥ 0, and = 0 **only when π = q** (Gibbs' inequality) | KL measures how different two distributions are. It's never negative, and only zero when they're identical |

E[...] means "the average of ... over samples." Here: prompts x from the dataset, and responses y from the policy π.

---

## Slide: "Start with the RLHF optimization problem"

```
max over π:   E_{x~D, y~π}[ r(x, y) ]  −  β · KL(π ‖ π_ref)
```

| Piece | Plain meaning |
|---|---|
| max over π | Find the best policy (the best set of probabilities over responses) |
| E[r(x, y)] | Average reward of the policy's answers |
| β · KL(π ‖ π_ref) | A penalty for drifting from the reference model. β sets how strong the leash is |

This is exactly the objective from `../lecture-3-policy-gradients/02-wiring-rlhf-for-language-models.md`. PPO and GRPO chase it by trial and error. "We want to find the π that solves this equation! But, without RL."

---

## Slides: "Fold the KL into the expectation" and "Flip to a minimization & clean-up"

Pure algebra. Each line is equal to the one before.

| Line | Move | Why |
|---|---|---|
| max E[ r − β·log(π/π_ref) ] | Write KL out as an average of log(π/π_ref) | Now everything is inside one average |
| max ( E[r] − β·E[log(π/π_ref)] ) | Split into two averages | Just bookkeeping |
| min ( −E[r] + β·E[log(π/π_ref)] ) | Multiply by −1: maximizing something = minimizing its negative | Minimization is the shape that KL / Gibbs' inequality speaks |
| min E[ log(π/π_ref) − (1/β)·r ] | Divide by β (a positive number doesn't change where the minimum is), recombine | Puts the log-ratio first, with reward as a correction |

Lambert on the sloppiness around which variables each E averages over: "the moving of expectations is kind of a minor artifact... it won't really bite you in the butt in the math."

---

## Slide: "Fold the optimization target into one log-ratio"

Use c = log(e^c) on the reward term: (1/β)·r = log(e^(r/β)). Then log a − log b = log(a/b):

```
min E[ log(π/π_ref) − log(e^(r/β)) ]  =  min E[ log( π(y|x) / (π_ref(y|x) · e^(r(x,y)/β)) ) ]
```

**What this says in words:** π is now measured against **π_ref × e^(r/β)**, "the reference reweighted by reward." Every response keeps its reference probability, multiplied by a boost:
- higher reward → bigger boost
- **β controls how aggressive the boost is.** Small β → e^(r/β) is huge for good answers, so the boost dominates. Large β → the boost is close to 1, so you stay near π_ref.

And this now **looks like a KL**: an average over π of log(π / something). If that "something" were a proper probability distribution q, the minimum would be at π = q.

---

## Slides: "Introduce the partition function Z(x)"

**The catch:** π_ref × e^(r/β) is **not a probability distribution.** Multiplying every probability by a boost means they no longer add up to 1. KL (and Gibbs' inequality) only works between real distributions.

**The fix:** divide by the total, to rescale. That total is the **partition function**:

```
Z(x) = Σ over all responses y of  π_ref(y|x) · exp( r(x, y)/β )
```

Then

```
q(y|x) = (1/Z(x)) · π_ref(y|x) · exp( r(x, y)/β )      ← adds up to 1: a real distribution
```

Two things to notice about Z(x):
- **It depends on the prompt x and the reward r, but not on π.** So it's a constant as far as "which π is best" goes.
- **It sums over every possible response.** For an LLM that's every possible string, an astronomically large number. **You can never compute it.** That's why the later steps work so hard to make it cancel (`03`).

Lambert's candid take: the partition function's definition "seems a little bit arbitrary... this is a step of the paper where you're kind of like, well, they kind of pulled that out of their ass. But that's life." It comes from established literature (statistical physics and energy-based models), and "this is how you do theoretical work... you draw on the kind of contributions of other areas."

### A tiny numeric example *(my addition)*

Pretend a prompt has only **3 possible responses**.

| Response | π_ref | reward r | boost e^(r/β), β = 1 | π_ref × boost |
|---|---|---|---|---|
| A | 0.5 | 0 | 1.00 | 0.500 |
| B | 0.3 | 1 | 2.72 | 0.815 |
| C | 0.2 | 2 | 7.39 | 1.478 |
| | sums to 1 | | | **Z = 2.793** (not 1!) |

Divide by Z to get q = π*: **A 0.179, B 0.292, C 0.529.** The reference's favourite (A) has been demoted, and the best answer (C) now gets over half the probability.

Same thing with **β = 10** (a strong leash): the boosts are only 1.00, 1.11, 1.22, Z = 1.076, and π* = **A 0.465, B 0.308, C 0.227**. Barely different from π_ref. That's β acting as the KL strength.

With 3 responses Z is easy. With all possible LLM outputs it's impossible, which is the whole problem.

---

## Slides: "Revisiting the objective with the partition function Z(x)"

Now get Z into the objective, without changing it. The "classic college math trick of adding zero":

| Line | Move |
|---|---|
| min E[ log(π/π_ref) − (1/β)r + log Z − log Z ] | Add 0 = log Z − log Z |
| min E[ (log(π/π_ref) + log Z) − log Z − (1/β)r ] | Regroup |
| min E[ log( π / ((1/Z)·π_ref) ) − log Z − (1/β)r ] | log a + log b = log(ab): the + log Z moves into the fraction as 1/Z in the denominator |

---

## Slide: "Turn the objective into a KL divergence"

Fold (1/β)·r into the denominator again (the same c = log e^c trick), and the denominator becomes exactly q:

```
min E_x E_{y~π}[ log(π(y|x)/q(y|x)) − log Z(x) ]
```

log Z(x) doesn't depend on y, so it comes out of the inner average:

```
min E_x [ E_{y~π}[ log(π/q) ]  −  log Z(x) ]
        = min E_x [ KL(π ‖ q)  −  log Z(x) ]
```

Lambert: "it seems like we kind of went through this twice, but it's good for repetition." The point of all the extra steps is "to be able to convince you that we can ignore this Z term."

---

## Slide: "Apply Gibbs' inequality to read off π*"

The objective is now **KL(π ‖ q) minus a constant that π can't affect.**

- log Z(x) doesn't depend on π, so only the KL can be changed.
- KL ≥ 0, and = 0 only when π = q (Gibbs' inequality).
- So the best possible π is q:

```
┌──────────────────────────────────────────────────────────┐
│  π*(y|x) = (1/Z(x)) · π_ref(y|x) · exp( r(x, y) / β )    │
└──────────────────────────────────────────────────────────┘
```

**In words:** the best RLHF policy is the reference model, with each answer's probability boosted by how good it is (e^(reward/β)), then renormalized.

Lambert: "this is actually the core of the DPO derivation." And the excitement: "holy crap. You can just do this. You can just go towards the optimal solution." DPO is "what is happening when you take the gradient from your data towards this optimal policy." He adds that it's "a general tool... just solving an RL problem with a direct gradient," so DPO-style approaches can also be used on verifiable reasoning problems.

*(My note: π* is the best policy in principle, over all possible distributions. A real neural network may not be able to represent it exactly, and the reward is only known through a finite dataset. DPO gets you close by gradient descent on preference pairs. The formula is the target, not something you compute.)*

---

## The whole section in four lines

1. **Start from the RLHF objective** (reward − β·KL) and rearrange it with log rules into "how far is π from the reference reweighted by e^(reward/β)?"
2. **That reweighted reference isn't a distribution**, so divide by the **partition function Z(x)**, a sum over every possible response. It's impossible to compute, but it doesn't depend on π.
3. **The objective becomes KL(π ‖ q) minus a constant**, and KL is smallest (zero) when π = q (Gibbs' inequality).
4. **So π* = π_ref · e^(r/β) / Z**: the reference, boosted toward high reward, with β as the leash. Next (`03`), turn this around to get the reward from the policy and make Z cancel.
