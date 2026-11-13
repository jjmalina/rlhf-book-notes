# Translating the Policy Gradient into Code

Notes on *Implementing RL Algorithms for LLMs | Post-Training Course, Lecture 4* (Nathan Lambert), 00:00–10:57 (intro + first section). Based on `source/lecture-4-implementing-rl-slides.md` and `source/lecture-4-implementing-rl-transcript.md`. Builds on all of `../lecture-3-policy-gradients/`. See `00-code-map.md` for where every tensor is computed and the PyTorch operations cheat sheet.

---

## Where this fits

Lecture 3 was the **math**. Lecture 4 is the **code**: how to implement, debug and run it.

Lambert's framing on the first slide:

> **"The hardest bugs aren't math errors. They're silent implementation mistakes: wrong masking, stale caches, shape mismatches."**

"Silent" means training runs without crashing, but the model learns the wrong thing. Much of this lecture is a list of places where that happens.

The good news: you already know the recipe. Every loss in this lecture is a variation on three lines you saw in lecture 3:

```python
seq_log_probs = (token_log_probs * completion_mask).sum(dim=-1)
loss = -(seq_log_probs * advantages).mean()
loss.backward()
```

This section takes those three lines apart: where `token_log_probs` comes from, what the mask does, why there's a minus sign, and how REINFORCE and RLOO compute `advantages`.

### A note on reading the code

The code is PyTorch. **Tensors** are multi-dimensional arrays, like NumPy arrays. Comments like `# (B, L)` give the **shape**:
- **B** = batch size (number of completions)
- **L** = sequence length (number of tokens)
- **V** = vocabulary size (~100k possible tokens)

So `(B, L)` means one number per token per completion. Every PyTorch operation used is explained in plain English in `00-code-map.md`.

---

## What "loss" means, and how the RL loss relates to the SFT loss *(my addition)*

A **loss** is **a single number that the training code tries to make as small as possible.** Training is a loop: compute the loss, call `backward()` to find which direction to move each weight to make it smaller, and step the weights that way. The **loss function** is the recipe for computing that number, and picking it is how you tell the optimizer what you want.

The word is used three ways:
- **loss function:** the formula (e.g. cross-entropy)
- **per-token loss:** one loss number per token, shape (B, L), before averaging (`06`)
- **the loss:** the single averaged number that `backward()` is called on

### SFT loss (cross-entropy)

SFT has the right answer, written by a person. For each of its tokens:

```
per-token loss = −log π(correct token)
```

| Model's probability for the correct token "4" | Loss = −log(p) |
|---|---|
| 0.05 | 3.00 |
| 0.30 | 1.20 |
| 0.90 | 0.11 |
| 1.00 | 0 (perfect) |

Minimizing it pushes the probability of the correct tokens up. It's the log-loss from logistic regression, applied to every token over a ~100k-word vocabulary.

```python
loss = -(token_log_probs * mask).sum() / mask.sum()     # SFT: average −log-prob of the human-written tokens
```

**The number means something:** how surprised the model is by the correct answer. It going down is real progress.

### RL loss (policy gradient)

There's no correct answer, only the model's own samples and their advantages:

```
per-token loss = −advantage × log π(sampled token)
```

```python
loss = -(advantages * seq_log_probs).mean()             # RL: advantage-weighted −log-prob of the model's own tokens
```

The same formula, with an extra multiplier. That's lecture 3's "weighted supervised learning," in code.

| | SFT loss | RL (policy gradient) loss |
|---|---|---|
| Tokens it's computed on | human-written answer | **model's own sampled answer** |
| Weight on each example | always 1 | **the advantage** (+ or −, any size) |
| Can push a token's probability *down*? | no, only toward the target | **yes**, when the advantage is negative |
| Does the number mean something? | yes: how well the model predicts the answer | **not really** |

Set advantage = 1 and use human-written text, and the RL loss becomes exactly the SFT loss.

### Why the RL loss number isn't meaningful

The SFT loss *measures* how good the model is. The RL loss is built so that **its gradient equals the policy gradient**, and its value is a by-product (lecture 3's "surrogate objective": a stand-in that points the right way). So it:
- **can be negative**, since advantages can be negative
- **jumps around**, because each batch has new samples and new advantages, unlike SFT's fixed dataset
- **isn't "how good the model is."** Reward going up is.

That's why `reward/mean` is the first panel in the monitoring table (`06`). The policy loss is still worth watching for **NaN or wild divergence**, which mean something broke, but **in RL, the reward is how you tell the model is learning**.

### PPO and GRPO losses

These use the ratio ρ instead of the log-prob: `per-token loss = −advantage × ρ` (with clipping). On the first step of a batch (ρ = 1) the gradient is identical to `−advantage × log π`, so it's the same family with a safety limit added.

PPO's total loss also adds the critic's own loss: `policy_loss + vf_coef × value_loss`. **That value loss *is* an ordinary SFT-style loss**: regression (MSE) toward the targets. That one does mean something.

---

## Intro slides (00:00–02:24)

- The RL chapter is "the crown jewel" of Lambert's book: the longest chapter and the one he's proudest of.
- Many real-world corner cases are about **data and scaling**, which are too specific to each codebase to cover fully.
- **Notation:** (s, a) state/action from RL, which reflects the per-token gradient. (x, y) prompt/completion, which reflects the per-sequence reward. Both appear.
- **Outline:** policy gradient code → PPO → GRPO & friends → training infrastructure → practical engineering → loss aggregation → conclusions.

---

## Slide: "Recall: The policy gradient"

```
∇J(θ) = E[ Σ_t  Ψ_t × ∇ log π_θ(a_t | s_t) ]
```

The slide's one-liner: **for each token, compute the direction that makes it more likely (∇ log π), then scale by how good it was (Ψ_t).** That's "which knobs × how good" from lecture 3.

## Slide: "Recall: The policy gradient algorithms"

All four algorithms from lecture 3, written as **losses** (with a leading minus, because code minimizes):

| Algorithm | Loss, in words |
|---|---|
| **REINFORCE** | −average over tokens of (log-prob × (return − baseline)) |
| **RLOO** | −average over K completions of (sum of log-probs × (reward − mean of the other K−1)) |
| **PPO** | −average over tokens of min(ρ × A, clip(ρ) × A), with A per token from GAE |
| **GRPO** | −average over completions of (average over tokens of min(ρ × Â, clip(ρ) × Â)), with Â = group z-score |

Lambert notes live that he mixed objectives and losses on this slide (some have the minus sign, some don't) and plans to fix it. The point stands: **they differ only in Ψ and in how updates are bounded**, and in code they look very similar.

## Slide: "Recall: Why losses are sums of log-probs"

From lecture 3's derivation: a completion's log-probability is the **sum** of its tokens' log-probs, so its gradient is the sum of per-token gradients. In code you compute the log-probs, and **autograd** (PyTorch's automatic differentiation, `loss.backward()`) produces the gradient. "Every loss function in this lecture is a variation on this pattern."

---

## Slide: "Computing log-probabilities (expanded)"

This is the building block every algorithm needs: **for each token the model actually wrote, what log-probability did the model give it?**

```python
# Forward pass through the model
logits = model(input_ids).logits              # (B, L, vocab_size)

# Autoregressive shift: logit at position t predicts token at t+1
logits = logits[:, :-1, :]                    # (B, L-1, vocab_size)
labels = input_ids[:, 1:]                     # (B, L-1)
completion_mask = completion_mask[:, 1:]       # (B, L-1)

# Per-token log-probs
log_probs = logits.log_softmax(dim=-1)
token_log_probs = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1) # (B, L-1)

# Per-sequence log-prob: sum over completion tokens
seq_log_probs = (token_log_probs * completion_mask).sum(dim=-1)
```

### Step by step, with an example

Take one sequence: prompt "What's 2+2?" and completion "4". As tokens:

```
position:   0        1    2    3    4    5     6
token:    What's     2    +    2    ?    4   <eos>
           ─────── prompt ───────────    ── completion ──
```

**1. Forward pass → `logits`, shape (B, L, V).** At every position the model outputs **a score for every token in the vocabulary** (~100k numbers), meaning "how likely is each possible next token?" These raw scores are called **logits**.

**2. The autoregressive shift.** The model's output at position t is a prediction for the token at position **t+1**. The output at "?" (position 4) is the model's guess for position 5, which turned out to be "4". So the two arrays have to be offset by one to line up:

```
logits[:, :-1]   = predictions made AT positions 0..5   (drop the last; nothing after it to predict)
input_ids[:, 1:] = actual tokens AT positions 1..6       (drop the first; nothing predicted it)

prediction made at "?"  ↔  actual token "4"     ✓ lined up
```

The mask gets the same shift so it stays aligned.

**3. `log_softmax` → log-probabilities over the vocabulary.** This turns each position's ~100k raw scores into ~100k log-probabilities (they'd sum to 1 as probabilities).

**4. `gather` → pick out the one you need.** You only care about the log-prob of **the token that was actually written**. `gather` uses `labels` as an index to pull out one number per position from the ~100k. The `unsqueeze`/`squeeze` calls just add and remove a size-1 dimension so the shapes fit. The result is `token_log_probs`, shape (B, L−1): one log-prob per token.

**5. Mask and sum → `seq_log_probs`, shape (B,).** The `completion_mask` is 1 for completion tokens and 0 for everything else (prompt, padding). Multiplying by it zeroes out the prompt, and summing gives **one log-prob per completion**.

**Why mask out the prompt?** The model didn't choose the prompt, it was given it. Training on "make the prompt more likely" is meaningless and wastes gradient. This comes up again as a silent bug.

**Why masks matter extra in practice:** batches often **pack** several sequences together, padded to the same length, to avoid idle compute. Masks tell the code which positions are real completion tokens, which are prompt, and which are empty padding.

**Why everything is in log-space:** multiplying many small probabilities together gives numbers too tiny for a computer to store accurately (**underflow**). Logs turn products into sums, which stay in a safe numerical range. (More in `05-practical-engineering.md`.)

Lambert admits his live explanation of the shift and gather could be clearer, and says to "go ask Claude or ChatGPT to explain it to you and challenge you on it." That's a reasonable thing to do with any line here.

## Slide: "Computing log-probabilities (compact)"

In real codebases this is wrapped in a helper you'll see everywhere:

```python
def compute_log_probs(model, input_ids, attention_mask):
    logits = model(input_ids=input_ids,
                   attention_mask=attention_mask).logits
    logits = logits[:, :-1, :].to(torch.float32)
    log_probs = F.log_softmax(logits, dim=-1)
    targets = input_ids[:, 1:].unsqueeze(-1)
    return torch.gather(log_probs, dim=-1,
                        index=targets).squeeze(-1)
```

It's the same shift, softmax and gather, condensed. Two details:
- `.to(torch.float32)` converts to full precision before the softmax, for numerical safety.
- It returns log-probs for **all** positions. Masking happens later.

**The timing part, which connects to `../lecture-3-policy-gradients/00-training-loop-timeline.md`:** this one function gets called on different models at different times:

| Called on | When | Result |
|---|---|---|
| The policy, at generation time | Once per batch (rollout) | **Old log-probs** (π_old). Cached. |
| The reference model | Once per batch (rollout) | **Ref log-probs** for KL. Cached. |
| The current policy | **Every training step** | **New log-probs**. Recomputed, because the weights keep changing. |

---

## Slide: "The REINFORCE loss in code"

```python
# rewards: (B,) — one reward per sequence
# seq_log_probs: (B,) — sum of log-probs over completion tokens

# Baseline: average reward in the batch
baseline = rewards.mean()
advantages = rewards - baseline

# REINFORCE loss (negative because we minimize)
loss = -(advantages * seq_log_probs).mean()
```

That's the entire algorithm. Mapped to lecture 3:

| Line | Lecture 3 idea |
|---|---|
| `baseline = rewards.mean()` | The simplest baseline: the average over the **whole batch** (across all prompts) |
| `advantages = rewards - baseline` | "Better or worse than expected?" |
| `advantages * seq_log_probs` | **Weight each completion's log-prob by its advantage** (weighted supervised learning) |
| `.mean()` | Average into one number, so `backward()` gives **one gradient per batch** |

### Why the minus sign?

Lambert pauses on this, saying it's "easy for people to lose track of":

> RL **maximizes** an objective (reward). Everything derived in lecture 3 points *uphill*. PyTorch optimizers **minimize** a loss (they go *downhill*). Minimizing −J is the same as maximizing J, so the minus sign flips it.

### A teaser

The slide flags that **how you reduce per-token losses to one number (`.mean()` here) "turns out to matter more than you'd expect."** That's the whole of `06-loss-aggregation-strategies.md`.

---

## Slide: "RLOO in code"

```python
# rlhf_reward: (B*K,) flat tensor of rewards
# Prompt-major layout: K sibling completions stay together
rlhf_reward = rlhf_reward.view(-1, rloo_k)  # (B, K)

# Leave-one-out baseline: avg of other K-1 rewards per prompt
baseline = (rlhf_reward.sum(dim=1, keepdim=True) - rlhf_reward) / (rloo_k - 1)

advantages = rlhf_reward - baseline        # (B, K)
advantages = advantages.reshape(-1)        # (B*K,)
```

(Here B = number of **prompts**, so B×K completions in total. Lambert notes the slide's letters are slightly inconsistent.)

### Line by line

**1. `view(-1, rloo_k)`: reshape the flat list into a grid**, one row per prompt and K columns for its completions. This only works if the K completions for each prompt sit **next to each other** in the flat list ("prompt-major layout"). The data loader has to guarantee that, or the rows mix up completions from different prompts. That's a silent bug waiting to happen.

```
flat:  [p1a, p1b, p1c, p1d, p2a, p2b, p2c, p2d, ...]
grid:  [[p1a, p1b, p1c, p1d],     ← prompt 1's 4 completions
        [p2a, p2b, p2c, p2d],     ← prompt 2's 4 completions
        ...]
```

**2. The leave-one-out trick.** Lambert calls it "kind of clever. It takes a good second to stare at it." To get "the average of everyone *except me*":
- sum the whole row (everyone, including me)
- subtract my own reward (removing me)
- divide by K−1

Using lecture 3's RLOO example, rewards [0.8, 0.3, 0.6, 0.5], row sum = 2.2:

| Completion | (row sum − mine) / 3 | Baseline | Advantage |
|---|---|---|---|
| 1 | (2.2 − 0.8)/3 | 0.467 | +0.333 |
| 2 | (2.2 − 0.3)/3 | 0.633 | −0.333 |
| 3 | (2.2 − 0.6)/3 | 0.533 | +0.067 |
| 4 | (2.2 − 0.5)/3 | 0.567 | −0.067 |

These match the lecture 3 table, computed for every completion at once in one line, with no loop.

`keepdim=True` keeps the sum as a column (shape (B, 1)) so it can be subtracted from every entry in its row. This is **broadcasting**: PyTorch automatically stretches the (B, 1) column across the K columns.

**3. `reshape(-1)`: flatten back** to one advantage per completion, so it lines up with `seq_log_probs` again.

**4. Then the loss is exactly REINFORCE's:** `loss = -(advantages * seq_log_probs).mean()`.

### REINFORCE vs. RLOO in code

The only difference is **one line**: the baseline.
- REINFORCE: `rewards.mean()`, one baseline for the whole batch (mixing easy and hard prompts, the "baseline problem" from lecture 3).
- RLOO: a per-prompt, leave-one-out baseline (handles prompt difficulty).

Lambert: "these initial policy gradient algorithms, the core loss is really, really fairly simple. It starts to get more complex as we go to something like PPO."

---

## The whole section in four lines

1. **Every RL loss is `−(advantages × summed token log-probs)` reduced to one number**, then `backward()`. The algorithms differ in how `advantages` is computed.
2. **Getting token log-probs:** run the model, **shift by one** (the output at t predicts token t+1), `log_softmax`, `gather` the actual token's log-prob, **mask** to completion tokens only, sum.
3. **The minus sign** turns "maximize reward" into "minimize loss," which is what optimizers do.
4. **REINFORCE vs. RLOO is one line:** batch mean vs. a leave-one-out mean per prompt (row sum − self, ÷ K−1). It needs completions grouped by prompt.
