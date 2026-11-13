# Policy Gradients: Core Intuitions and Maths

Notes on *Understanding Policy Gradient Algorithms for RL on LLMs | Post-Training Course Lecture 3* (Nathan Lambert), 06:04–25:38. A slide-by-slide breakdown, based on `source/lecture-3-policy-gradients-slides.md` and `source/lecture-3-policy-gradients-transcript.md`. Slides that appear twice because they reveal more on a click are merged.

---

## The one idea to hold onto

Policy gradients answer one question: **after the model writes something and gets a score, how do we nudge each weight?**

The whole answer is:

> **For every token the model generated, push the weights to make that token more likely if the outcome was good, and less likely if it was bad, scaled by how good or bad.**

Everything in this section either proves that this simple rule is mathematically correct, or makes it less noisy.

One example is used throughout. Prompt: *"What's 2+2?"* The model samples 4 answers, and the reward model scores them:

| Completion | Reward |
|---|---|
| "4" | 1.0 |
| "The answer is 4" | 0.9 |
| "5" | 0.0 |
| "fish" | 0.0 |

---

## How it works, and why it's an estimate

Going from the current weights and the rewards to the policy gradient is an estimate. But only one part of it is estimated, and seeing which part makes the "how" clearer.

### The core problem: you can't backprop through the reward

In SFT there's a target answer, so the loss is differentiable: backprop runs from the loss, through the model, to every weight.

In RL the chain looks like this:

```
weights → token probabilities → SAMPLE a token → ... → full completion → reward model → score
                                 ^^^^^^^^^^^^^^
                                 discrete choice, no gradient flows through here
```

Picking a token is a dice roll. "If I nudged weight #5,000,123, how would the reward change?" has no derivative through that roll. So you can't compute "how the weights affect reward" directly.

### The workaround: differentiate the probability, use the reward as a weight

The model can't tell how its weights affect the *reward*. It can tell exactly how its weights affect *the probability of the tokens it picked*. So you split the problem into two parts:

**Step 1 (exact, same as SFT): "which knobs make this token more likely?"**
For each generated token, backprop on `log π(token)`. For every weight, that gives the direction that would make that exact token a bit more likely. Nothing is estimated here. It's the same backprop as SFT, as if the model's own output were the training target.

**Step 2: "should we actually do that, and how much?"**
Multiply that direction by Ψ (reward minus baseline):
- Ψ = +0.525 for "4": go in that direction, fairly strongly. "4" becomes more likely.
- Ψ = −0.475 for "fish": go the *opposite* way. "fish" becomes less likely.
- Ψ = 0: skip it.

**Step 3: add everything up**
Sum over every token in every completion, then average. Directions that most good samples agree on survive. Directions pulling against each other cancel out. That net direction is the policy gradient.

There's a bonus here. Because probabilities must sum to 1, raising "4" automatically lowers everything else, and lowering "fish" frees up probability for other tokens.

### Where the estimation comes in

The **true** policy gradient is that sum taken over *every possible completion*, each weighted by how likely the model is to produce it. That's an astronomically large number of completions, so you sample 4–64 per prompt and use those instead. It works like a poll:

- **It's noisy.** A different batch of samples gives a somewhat different gradient.
- **It's unbiased.** On average it points the right way, because likely completions show up in your samples more often. That's the "p_θ(τ) became the sampling distribution" slide.
- **Ψ is often estimated too.** The baseline is a guess at "expected reward": the group mean in GRPO/RLOO, or a learned value model in PPO.
- **Credit assignment is crude.** Every token in "The answer is 4" gets the same +0.425, including "The", which didn't really earn it. That averages out across many samples, because "The" shows up in good and bad answers alike, so its pushes cancel.

That's why the lecture spends so much time on variance. The math guarantees you're pointing the right way *on average*. The baseline and advantage tricks reduce how much each individual estimate wanders.

### One-line version

> You can't measure how the weights affect the reward, so you measure how they affect the probability of what the model did, then reinforce or suppress that in proportion to the reward. Do this across a sample of completions and the average is a noisy but correct-on-average estimate of the true gradient.

---

## Part 1: Framing (~06:04–09:00)

### Slide: "Classical RL vs. RLHF"
- **Classical RL** (robots, Atari): an agent acts in an unknown world, and the world hands back rewards at each step.
- **RLHF/RLVR**: there's no unknown world. We pick the prompts and we build the reward function, either a reward model or a verifier like "is the answer correct?"
- The `β·KL` term in the RLHF objective is a leash that stops the model drifting too far from where it started. It isn't covered in this section, so ignore it for now.

**Takeaway:** for LLMs we design the environment, so a lot of the classical RL complexity goes away.

### Slide: "Notation"
- RL papers write **s** (state) and **a** (action). LLM papers write **x** (prompt) and **y** (completion). They describe the same thing at different granularity.
- **τ (tau)** = a trajectory = one prompt plus one full completion.
- **r** = reward and **ρ (rho)** = a probability ratio (comes later, for PPO).

### Slide: "Language models as episodic MDPs"
This table translates RL terms into LLM terms:
- **State** = prompt + tokens written so far
- **Action** = the next token
- **Policy π_θ** = the language model. **θ** = its weights.
- **Transition** = append the token. Nothing random happens here, which matters later.
- **Episode** = one completion
- **γ (discount)** = 1, meaning we don't discount future rewards.

**Takeaway:** writing a completion is a sequence of decisions, one per token. That's the only reason RL applies here.

---

## Part 2: The intuition (~09:00–15:30)

### Slide: "What are we optimizing?"
```
J(θ) = E_τ~p_θ [ R(τ) ]
```
In plain words: **J is the average reward the model gets when it writes completions.** We want weights θ that make J as high as possible.

We can't average over every possible completion, so we sample a handful and average those. In the example: (1.0 + 0.9 + 0 + 0)/4 = 0.475.

The key problem: we know how to measure J, but we need its **gradient**, the direction to move each weight. And the reward model's score doesn't depend on the weights in a way we can differentiate. It's just a number handed to us.

### Slide: "How a policy gradient works" ⭐ the most important slide
```
Δθ ∝ Ψ_t · ∇_θ log π_θ(a_t | s_t)
```
There are two pieces, and each answers one question:

1. **`∇_θ log π_θ(a_t | s_t)` answers "which knobs?"**
   This is the gradient of the log-probability of the token the model actually picked. It tells you which direction to move every weight to make that exact token more likely. It's the same quantity you compute in ordinary supervised fine-tuning (SFT) or pretraining when you call `loss.backward()` on cross-entropy.

2. **`Ψ_t` (psi) answers "how good was it?"**
   This is one number. Positive means "do more of this," negative means "do less," and its size says how much.

Multiply them together and you get: "make this token more likely, scaled by how good the outcome was."

**Connection to SFT:** if Ψ = 1 for every token, this is exactly SFT on the model's own outputs. Policy gradient is SFT where each example is weighted by its reward. Good completions get trained on, and bad ones get *un*-trained (when Ψ is negative).

### Slide: "The policy gradient equation" (preview)
```
∇J(θ) = E[ Σ_t  Ψ_t · ∇ log π_θ(a_t | s_t) ]
```
This is the same thing as the previous slide, summed over every token in the completion and averaged over samples. **The rest of the lecture is about choosing Ψ well.**

### Slide: "Key RL quantities"
These are three candidate answers to "how good was it?":
- **V(s)**, the value: how much reward you expect from here on, on average. ("Halfway through this answer, how well is it going?")
- **Q(s, a)**, the action-value: how much reward you expect if you pick this specific token now.
- **A(s, a) = Q − V**, the advantage: how much better this token was than average. This is the most useful one. +0.3 means better than usual, −0.3 means worse than usual, and 0 means don't change anything.

### Slide: "What Ψ_t can be" (the table from Schulman et al., 2015)
This is a menu of choices for Ψ, ordered from simplest and noisiest down to smartest:
1. **Total reward of the whole completion.** Simple but very noisy.
2. **Reward from this token onward.** Drops rewards from earlier steps, which this token couldn't have caused.
3. **Reward minus a baseline.** Compares against "what we expected."
4. **Q**
5. **Advantage.** Lowest noise if you can estimate V well.
6. **TD residual.** A one-step estimate (used in PPO, later lecture).

"Variance" here means **noise in the gradient**: how much the update jumps around from batch to batch. "Bias" means **systematically pointing the wrong way**. You want low noise with no bias.

---

## Part 3: The derivation (~15:30–25:38)

This part proves that the "which knobs × how good" rule is actually the correct gradient of J. Here's the logic without the calculus.

### Slide: "Setup: Differentiating an expectation"
```
J(θ) = ∫ p_θ(τ) R(τ) dτ
∇J  = ∫ ∇p_θ(τ) R(τ) dτ
```
- Line 1 is a definition: average reward = the sum over all completions of (probability of the completion × its reward).
- Line 2: the reward R doesn't depend on θ (the reward model is fixed). The only thing the weights change is **how likely each completion is**. So the gradient lands entirely on the probability p_θ.
- **Problem:** `∇p_θ(τ)` isn't a probability distribution, so we can't estimate it by sampling. We need to rewrite it.

### Slide: "Log-derivative identity" (the one trick)
```
∇ log p = ∇p / p     →     ∇p = p · ∇ log p
```
This is just the chain rule (the derivative of log(x) is 1/x), rearranged. Why bother? Because it pulls a plain `p` back out front:

```
∇J = ∫ p_θ(τ) · R(τ) · ∇log p_θ(τ) dτ  =  E[ R(τ) · ∇log p_θ(τ) ]
```

### Slide: "Where did p_θ(τ) go?"
"∫ p(τ) × something" is by definition an average over samples from p. So you don't compute p anywhere in the code. **You sample completions from the model, and likely completions show up more often on their own.** The weighting by p happens through sampling.

So: generating rollouts handles the "p" part, and the loss function handles the "R × ∇log p" part.

### Slides: "Expanding the log probability of a trajectory"
The probability of a whole completion is a product:
```
p(τ) = p(prompt) × π(token1) × π(token2) × ... × (transition terms)
```
Take the log and the product becomes a sum. Then take the gradient with respect to the weights:
- `p(prompt)`: the model didn't choose the prompt, so its gradient is **0**
- transition terms: "append the token" doesn't involve the weights, so their gradient is **0**
- what's left is **Σ ∇ log π(token_t)**

So the gradient of a completion's log-probability is **the sum of the per-token log-prob gradients**, the same quantity you already compute in normal LM training.

That's why the code is only three lines:
```python
seq_log_probs = (token_log_probs * completion_mask).sum(dim=-1)   # sum of log π over the completion's tokens
loss = -(seq_log_probs * advantages).mean()                        # weight by Ψ, average over the batch
loss.backward()                                                    # PyTorch computes ∇ for you
```
The minus sign is there because optimizers *minimize*, and we want to *maximize* reward. The "where is the gradient?" comment in the lecture refers to this: you never write ∇ yourself. You build "log-probs × reward" and autograd handles the rest.

### Slide: "From full return to future rewards"
A token at position 10 can't affect a reward given at position 5. So weight each token by **only the rewards from that point on** (G_t, the "return-to-go"). The average gradient stays the same, and you remove noise. This is moving from menu option 1 to option 2.

(For LLMs with a single reward at the end, options 1 and 2 are identical. This matters more in classical RL.)

### Slides: "Baselines keep the estimator unbiased" ⭐ second most important idea
**The problem:** look at the example again. Rewards are 1.0, 0.9, 0, 0, all ≥ 0. With raw rewards as Ψ, the "fish" answer gets weight 0, so it's ignored rather than pushed down. If every reward were positive (say 5, 6, 7, 8), the update would make *every* answer more likely, just by different amounts. That works on average, but it's very noisy.

**The fix:** subtract a baseline, "what we expected":
```
Ψ = reward − baseline
```
With baseline = mean reward = 0.475:

| Completion | Reward | Ψ = reward − 0.475 | Effect |
|---|---|---|---|
| "4" | 1.0 | **+0.525** | push up |
| "The answer is 4" | 0.9 | **+0.425** | push up |
| "5" | 0.0 | **−0.475** | push down |
| "fish" | 0.0 | **−0.475** | push down |

Now the question changes from "was this good?" to **"was this better than usual?"** That gives a much cleaner signal.

**Why this is allowed (the math on the third slide):** subtracting the baseline adds a term `E[b · ∇log π]`. Because the model's probabilities over all tokens always sum to 1, their gradients sum to 0 (you can't raise every token's probability at once). So the extra term averages to **exactly zero**:
```
E[ b(s) · ∇log π(a|s) ] = b(s) · Σ_a ∇π(a|s) = b(s) · ∇ Σ_a π(a|s) = b(s) · ∇(1) = 0
```
The baseline lowers noise without changing where the gradient points on average. The one condition is that the baseline can't depend on which action was chosen, only on the state or prompt.

This is the foundation of GRPO and RLOO: **the baseline is the average reward of the other samples for the same prompt.** That's what this example did.

### Slides: "The policy gradient estimator" + "Derivation recap"
Putting it all together:
```
∇J(θ) = E[ Σ_t  Ψ_t · ∇ log π_θ(a_t | s_t) ]
```
This is the **policy gradient theorem**. Every algorithm that follows (REINFORCE, RLOO, PPO, GRPO) uses this same formula with a different recipe for Ψ.

---

## The whole section in four lines

1. **Goal:** raise the average reward J of the model's completions.
2. **Trick:** the log-derivative identity turns "gradient of an average" into "average of (reward × gradient of log-prob)," which you can estimate by sampling.
3. **Simplification:** a completion's log-prob is a sum of token log-probs, so the gradient is a sum of per-token gradients, the same ones as in SFT.
4. **Noise reduction:** replace raw reward with "reward minus expected reward" (a baseline or advantage). The gradient points the same way on average and is much less noisy.

**Training loop in practice:** sample several completions per prompt → score them → subtract the average score → compute loss = −(advantage × sum of token log-probs) → `backward()` → step.

---

## Clarification: the policy gradient vs. the weight update

The policy gradient is the **direction** the weights should move. The update is the step you actually take in that direction. They're separate:

```
policy gradient:   g = ∇J(θ)            ← "which way is uphill for reward" (one number per weight)
weight update:     θ ← θ + lr · g       ← actually moving the weights
```

In practice:
- **`loss.backward()`** computes the policy gradient and stores it in each parameter's `.grad`.
- **`optimizer.step()`** does the update. With Adam, which is what people actually use, the step is the gradient reshaped by momentum and per-weight scaling, not just `lr × gradient`.

So the policy gradient is the ingredient and the optimizer turns it into the update. The "policy gradient" part of the name refers to how that direction gets computed. Everything after it is ordinary deep learning, the same optimizer you'd use for supervised fine-tuning.

This doesn't happen once after training. It repeats every batch:

1. Sample completions from the current model.
2. Score them.
3. Compute the policy gradient (`backward()`).
4. Update the weights (`step()`).
5. Go back to 1 with the slightly improved model.

Each update changes the model a little, so the next batch of completions is sampled from a slightly different model. That loop is what makes it reinforcement learning rather than a single training pass on fixed data.

---

## One policy gradient per batch, not one per sample

A batch is **many prompts × N completions each** (e.g. 64 prompts × 4 completions = 256). All the completions in the batch are combined into a **single** gradient, which is applied once.

There are two scopes, and they're easy to blur:

| | Scope | What happens |
|---|---|---|
| **Advantage** | **Per prompt** | A prompt's N completions are compared with each other to get one advantage per completion |
| **Gradient** | **Per batch, across all prompts** | Every completion's weighted push is averaged into one gradient and one update |

```
batch = 64 prompts × 4 completions

prompt 1:  "What's 2+2?"        4 completions → compare within group → 4 advantages
prompt 2:  "Capital of France?" 4 completions → compare within group → 4 advantages
...
prompt 64: "Write a haiku"      4 completions → compare within group → 4 advantages
                                                  │
                         all 256 (advantage × which-knobs) → average → ONE gradient → ONE update
```

**What gets averaged is not the advantages.** Each completion has its own "which knobs" vector (how to nudge every weight to make that answer more likely). Its advantage scales that vector up, down or flips it. The gradient is the **average of those scaled vectors**:

```
✗  gradient = average(advantages)                      ← just a number, and ≈ 0 since advantages are centered
✓  gradient = average(advantage × which-knobs vector)  ← one vector, one value per weight
```

Why not one gradient per prompt? You *could*, but it's the same reason supervised learning uses minibatches: a gradient from one prompt is very noisy and tuned to that one question, averaging over many prompts keeps what they share ("be correct", "be clear"), and GPUs process many prompts in parallel.

The loop:

```
for step in range(many_steps):                          ← the outer loop repeats thousands of times
    sample prompts, generate N completions per prompt   ← the batch
    grade each completion                               ← rewards
    advantage = reward − baseline (e.g. group mean)     ← "better or worse than usual?"
    loss = −(advantage × sum of token log-probs).mean() ← .mean() combines every completion in the batch
    loss.backward()                                     ← ONE policy gradient (one value per weight)
    optimizer.step()                                    ← ONE update
```

Each completion contributes its own "push" (which knobs × how good). The `.mean()` averages all those pushes into **one** policy gradient, and that average is the estimate. More samples give a better estimate, the same way a poll of 1,000 people beats a poll of 4. Updating separately for each sample would mean acting on a 1-person poll every time.

Notes:
- **"A bunch of math" is really two lines of code.** The derivation exists to *prove* that `−(advantage × log-probs).mean()` followed by `backward()` gives the correct gradient. At training time, autograd does all of the work.
- **Rewards don't have to be 0.0–1.0.** Verifiers often give 0/1, but reward models output any real number. The baseline puts rewards on a common scale.
- *(Preview, later in the lecture)* PPO is an exception: it takes one batch and runs **several** gradient updates on it (minibatches, multiple passes) to get more learning out of each expensive round of generation. That's why PPO needs the ratio ρ and clipping.

---

## Subtracting the expected reward (the baseline)

"Expected reward" means **how well the model usually does on this prompt.** Subtracting it changes the question from "was this answer good?" to "was this answer better or worse than the model's usual answer?" This quantity, reward − expected reward, is the **advantage**.

### How you get the expected reward

The section only says "subtract some b(s)." The simplest concrete recipe (RLOO/GRPO, *covered after 25:38*) is to take the average reward of the N completions sampled for that same prompt:

```
"What's 2+2?" → 4 samples → rewards 1.0, 0.9, 0.0, 0.0
expected reward ≈ average = 0.475
```

PPO trains a separate value model to predict this number instead, but the idea is the same.

### Why subtract it

**Problem 1: bad answers don't get pushed down.**

| Completion | Raw reward used as Ψ | Ψ = reward − 0.475 |
|---|---|---|
| "4" | 1.0 → push up | +0.525 → push up |
| "The answer is 4" | 0.9 → push up | +0.425 → push up |
| "5" | 0.0 → **ignored** | −0.475 → **push down** |
| "fish" | 0.0 → **ignored** | −0.475 → **push down** |

With raw rewards, wrong answers never get actively discouraged. After subtracting, they do.

**Problem 2: everything gets pushed up when rewards are all positive.**

Say a reward model scores four answers 6, 7, 8, 9. Used raw, all four get pushed up, the weakest answer included, just by slightly different amounts. Subtracting the average of 7.5 gives −1.5, −0.5, +0.5, +1.5. Now the two worse answers go down and the two better ones go up.

**Problem 3: easy and hard prompts get treated unfairly.** *(this is the "baseline problem" slide in `03`)*

- On an easy prompt, the model gets 1.0 almost every time. Another 1.0 isn't news. 1.0 − 1.0 = 0, so nothing changes, and that's correct because there's nothing to learn.
- On a hard prompt, the model usually gets 0.1. A 0.6 is a big win. 0.6 − 0.1 = **+0.5**, a strong push.

Without the baseline, the easy prompt's 1.0 would produce a bigger push than the hard prompt's 0.6, which gets things backwards.

### Why this is allowed

It looks like changing the rewards would change what the model learns. The baseline slides prove that it doesn't change the average direction of the gradient, as long as the baseline depends only on the prompt and not on which answer was picked. It only makes each estimate less noisy.
