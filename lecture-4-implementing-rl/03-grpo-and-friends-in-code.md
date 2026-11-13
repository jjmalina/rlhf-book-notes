# GRPO & Friends in Code

Notes on *Implementing RL Algorithms for LLMs | Post-Training Course, Lecture 4* (Nathan Lambert), 33:30–38:51. Based on `source/lecture-4-implementing-rl-slides.md` and `source/lecture-4-implementing-rl-transcript.md`. Follows on from `02-ppo-implementation.md`. The concepts are in `../lecture-3-policy-gradients/05-grpo-and-modern-variants.md`, and this file is their code.

---

## Where this fits

After PPO, Lambert's relief is audible: GRPO and friends "are much, much simpler to implement, and that is wonderful."

Compared with `02`, GRPO's code **deletes** the critic, GAE, value targets, value clipping and value initialization, and replaces all of it with **about five lines of group statistics**. The clipped policy loss is copied from PPO almost unchanged.

---

## Slide: "Recall: GRPO"

For each prompt, sample G completions:

```
Â_i = (R_i − μ_G) / σ_G        μ_G = group mean,  σ_G = group std
```

Then the PPO clipped loss with **per-token ratios but one advantage per completion**, plus an optional KL term:

```
L_GRPO = −(1/G) Σ_i (1/|a_i|) Σ_t min(ρ_{i,t} Â_i, clip(ρ_{i,t}) Â_i)  +  β·KL(π_θ ‖ π_ref)
```

"No value function, no GAE. Advantages come entirely from comparing siblings within a group."

### Lambert's "exercise for the reader": should it be +β·KL or −β·KL?

He thinks the slide is right and leaves it as an exercise. **It's right**, and here's why:
- This is a **loss** (minimized), with a leading minus on the policy part.
- You want to **penalize** drift from the reference, which means **adding** KL makes the loss worse.
- In lecture 3 `05`, the same thing was written as an **objective** (maximized) with **−β·KL**. Flipping every sign to turn the objective into a loss turns −β·KL into **+β·KL**.

It's the same sign-flip logic as the minus sign in `01` and the `torch.max` in `02`.

---

## Slide: "GRPO in code"

```python
# Generate G completions per prompt, then compute group advantages
mean_r = rewards.view(-1, G).mean(dim=1)
std_r = rewards.view(-1, G).std(dim=1)
mean_r = mean_r.repeat_interleave(G)
std_r = std_r.repeat_interleave(G)
advantages = ((rewards - mean_r) / (std_r + 1e-4)).unsqueeze(1)

# Importance sampling ratio
ratio = torch.exp(new_logps - old_logps)   # (B*G, L)

# Clipped surrogate (same as PPO)
pg_losses1 = -advantages * ratio
pg_losses2 = -advantages * torch.clamp(ratio, 1 - eps, 1 + eps)
pg_loss = torch.max(pg_losses1, pg_losses2)

# KL penalty in loss (not in reward)
per_token_loss = pg_loss + beta * per_token_kl

loss = ((per_token_loss * mask).sum(dim=1) / mask.sum(dim=1)).mean()
```

### Block 1: group advantages (this replaces all of GAE)

| Line | What it does |
|---|---|
| `rewards.view(-1, G)` | Reshape the flat rewards into one row per prompt (G columns), like RLOO in `01`. It needs completions grouped by prompt. |
| `.mean(dim=1)`, `.std(dim=1)` | Each row's mean and spread: **one mean and one std per prompt** |
| `.repeat_interleave(G)` | Copy each prompt's mean/std G times so it lines up with the flat list again: [μ₁, μ₁, μ₁, μ₁, μ₂, μ₂, …] |
| `(rewards − mean_r) / (std_r + 1e-4)` | The z-score: one advantage per completion |
| `+ 1e-4` | **The fix for the zero-std bug** from `02`. If all G rewards are equal, std = 0, and this avoids NaN (the advantage just becomes 0) |
| **`.unsqueeze(1)`** | Turns shape (B·G,) into (B·G, 1). **This is "copy the advantage to every token."** When it's multiplied by `ratio`, shape (B·G, L), broadcasting stretches each completion's single advantage across all L of its tokens |

That last row is worth pausing on. In lecture 3, "the sequence-level advantage is broadcast to every token" sounded abstract. **In code it's one `unsqueeze(1)` plus PyTorch broadcasting.**

**A small detail** *(my addition)*: PyTorch's `.std()` divides by G−1 by default (the "sample" std), while the slide's formula divides by G. For [1, 0, 0, 0] that gives advantages **+1.5 / −0.5** instead of lecture 3's **+1.73 / −0.58**. It's the same direction and ranking, only the scale differs, and whitening and learning rates absorb that.

### Block 2: ratio and clipped loss (copied from PPO)

These are the same three lines as `02`'s PPO policy loss, using `exp(new − old)` and `torch.max`. The only difference is that `advantages` is one value per completion, broadcast, instead of GAE's per-token values.

### Block 3: KL in the loss

`per_token_loss = pg_loss + beta * per_token_kl`. Where PPO added `vf_coef × vf_loss` (the critic), **GRPO adds the KL term instead**. There's no critic loss at all.

### Block 4: aggregation

Average within each sequence, then across sequences. This is the per-sequence normalization from `06`.

---

## Slide: "KL penalty placement"

The same goal (constrain drift from the reference), in a different place:

**PPO: KL in the reward**
```python
# per_token_kl is (B, L), rewards is (B, L)
# Scatter RM score to each sequence's
# last action token, not [:, -1]
rewards.scatter_(1, last_idx, rm_score)
rewards = rewards - beta * per_token_kl
advantages = gae(rewards, values, ...)
```

**GRPO: KL in the loss**
```python
# Compute advantages from raw rewards
advantages = z_score(rewards)
# Add KL as separate loss term
loss = pg_loss + beta * per_token_kl
```

### Why they differ (from the transcript)

- **PPO was designed RLHF-first, with a critic.** The critic gives per-token credit, so it's natural to make KL part of the per-token rewards that GAE processes.
- **GRPO broadcasts one advantage to every token anyway.** Folding a per-token KL into a per-sequence z-score would muddle it. Lambert: "if you're already broadcasting the advantages, might as well just apply the KL directly to the loss, which is also per token, and there's less transformations."
- The slide's reason: **GRPO's placement "avoids interaction between KL and advantage estimation."** The advantage stays a clean "better or worse than siblings" signal.

**Timing** (matches `00-code-map.md` and lecture 3's timeline): PPO's KL is used **at rollout**, before advantages. GRPO's KL is used **in the loss, at every training step**.

Lambert: "You could do both of these in different ways if you wanted to. These algorithms are fairly robust to tinkering. It's just good to know what the actual defaults are."

Note the comment in the PPO code: **`last_idx`, not `[:, -1]`**. It's the reward-placement silent bug from `02` again.

---

## Slide: "GRPO vs PPO implementation comparison"

What GRPO removes:

| Component | PPO | GRPO |
|---|:---:|:---:|
| Value network | Required | **None** |
| GAE computation | Required | **None** |
| Critic loss | Required | **None** |
| Value function init (and warmup) | Required | **None** |
| Advantage computation | Per-token (GAE) | Per-sequence (z-score) |
| KL handling | Folded into reward | Separate loss term |

"Significantly less code and ~1 fewer model copy in memory." Lambert: "there's a lot fewer moving parts in GRPO, which has made tinkering and research much easier."

---

## Slide: "RLOO vs GRPO in code"

```python
# RLOO: leave-one-out mean
rewards = rewards.view(-1, K)         # (N, K)
baseline = (rewards.sum(dim=1, keepdim=True) - rewards) / (K - 1)
advantages = rewards - baseline
advantages = advantages.reshape(-1)
```

```python
# GRPO: group z-score normalization
rewards = rewards.view(-1, G)       # (N, G)
mean_r = rewards.mean(dim=1, keepdim=True)
std_r = rewards.std(dim=1, keepdim=True)
advantages = (rewards - mean_r) \
             / (std_r + 1e-4)       # (N, G)
advantages = advantages.reshape(-1)
```

The same structure: reshape to (prompts × group), compute a per-row baseline, subtract, flatten. The differences:

| | RLOO | GRPO |
|---|---|---|
| Baseline | mean of the **others** (row sum − self) / (K−1) | mean of the **whole row** |
| Scaling | none | **÷ std** |
| Then | REINFORCE loss (no ratio, no clip) | PPO-style clipped loss |

(This version uses `keepdim=True` + broadcasting instead of `repeat_interleave`. Both are the same operation.)

Lambert: "it's interesting to see these ideas recur, especially simple ones, in the RL literature."

---

## Slides: "Recent advancements: GSPO & CISPO" + code

Both are about **numerical stability at scale**. A paper called ScaleRL found CISPO "very useful," and Lambert says both "seem to be very important for doing at-scale models."

### GSPO: one ratio per sequence (a GRPO-style algorithm)

```python
log_ratio = (new_logps - old_logps) * mask
rho = torch.exp(log_ratio.sum(dim=1) / mask.sum(dim=1))  # (B*G,)
# Same clipped loss as GRPO, but per-sequence
rho_clipped = rho.clamp(1 - eps, 1 + eps)
loss = -torch.min(rho * advantages, rho_clipped * advantages).mean()
```

| Line | What it does |
|---|---|
| `(new − old) * mask` | Per-token log-ratios, completion tokens only |
| `.sum(dim=1) / mask.sum(dim=1)` | **Average log-ratio per sequence** |
| `torch.exp(…)` | Back to a ratio. This is the **geometric mean** from lecture 3, giving one ρ per completion, shape (B·G,) |
| clamp + min | The same clipped surrogate, applied once per sequence instead of once per token |

Lambert's summary: "you take the log ratio, sum over it at each token, and then you exponentiate it. So instead of having ρ be per token, you pull it all out into one value."

(Note that this one uses `torch.min` with **positive** `rho * advantages` inside a leading minus, so it computes min of the objective then negates, instead of max of negated losses. It's the same math as PPO's `torch.max` version, written the other way round.)

### CISPO: stop-gradient on the clipped ratio (a REINFORCE-style algorithm)

```python
rho = torch.exp(new_logps - old_logps)
rho_clipped = torch.clamp(rho, 1 - eps, 1 + eps).detach()  # no grad through ratio
loss = -(rho_clipped * advantages.unsqueeze(1) * new_logps * mask).sum() / mask.sum()
```

| Line | What it does |
|---|---|
| `torch.clamp(…)` | Cap the ratio to the band |
| **`.detach()`** | **This is the "stop-gradient" (sg) from lecture 3.** The capped ratio becomes a fixed multiplier, and no gradient flows through it |
| `rho_clipped * A * new_logps` | **REINFORCE's `advantage × log-prob`**, with a capped weight in front. The gradient flows only through `new_logps` |
| `.sum() / mask.sum()` | Per-**token** aggregation (`06`) |

The key behavioral difference from PPO/GRPO, in code terms:
- **PPO/GRPO:** when ρ leaves the band, `torch.max` picks the clamped branch, which is a **constant** with respect to the weights, so **the token's gradient becomes 0**.
- **CISPO:** the gradient always flows through `new_logps`. The clamped ρ only **scales** it. **Every token always gets some gradient.**

That's lecture 3's "off switch vs. cap," visible in one line: whether the ratio is inside the gradient path, or `.detach()`ed out of it.

---

## The whole section in four lines

1. **GRPO's advantage code is ~5 lines:** `view(-1, G)` → per-row mean/std → z-score (with `+1e-4` to avoid NaN) → **`unsqueeze(1)` broadcasts one advantage to every token**.
2. **The policy loss is PPO's, copied.** Where PPO adds `vf_coef × value_loss`, GRPO adds `β × per_token_kl`. KL moves from the reward (computed at rollout) to the loss (every step), which keeps the advantage clean. The **+β·KL** is correct, because it's a loss.
3. **RLOO vs. GRPO in code:** the same reshape-baseline-flatten shape. Leave-one-out mean vs. group mean ÷ std, then a REINFORCE loss vs. a clipped loss.
4. **GSPO** = average the log-ratios per sequence, then exp, for one clip per sequence. **CISPO** = `clamp(ρ).detach() × A × log-prob`, REINFORCE with a capped weight, where every token always gets gradient.
