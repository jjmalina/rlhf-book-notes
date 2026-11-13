# PPO Implementation

Notes on *Implementing RL Algorithms for LLMs | Post-Training Course, Lecture 4* (Nathan Lambert), 10:57–33:30. Based on `source/lecture-4-implementing-rl-slides.md` and `source/lecture-4-implementing-rl-transcript.md`. Follows on from `01-translating-the-policy-gradient-into-code.md`. The concepts are in `../lecture-3-policy-gradients/04-ppo.md`, and this file is their code. See `00-code-map.md` for the tensor timeline and the PyTorch cheat sheet.

---

## Where this fits

This is the longest and hardest section of the lecture (Lambert: "the hardest part of this lecture is actually done" when it ends). PPO has the most moving parts:

| Lecture 3 concept | Code in this section |
|---|---|
| Per-token rewards (KL + final score) | Reward shaping |
| Critic + GAE → per-token advantages | The GAE loop (runs **backward** over tokens) |
| Ratio ρ + clipping | `exp(new − old)`, `clamp`, `torch.max` |
| Several steps per batch | Epochs and **minibatches** (new: minibatching alone can trigger clipping) |
| Critic trained by regression | **Value targets** and a **clipped** value loss (new) |
| Two losses at once | `policy_loss + vf_coef × value_loss` |

Plus a stack of practical topics: advantage whitening, value initialization, hyperparameters, why batch size matters so much, a debugging checklist and memory costs.

---

## Slide: "Recall: PPO"

The clipped loss (minimized) and GAE, the same as lecture 3 `04`:

```
L_CLIP = −E[ min( ρ_t A_t , clip(ρ_t, 1−ε, 1+ε) A_t ) ]
A_t^GAE = Σ_l (γλ)^l δ_{t+l},    δ_t = r_t + γ V(s_{t+1}) − V(s_t)
```

λ = 0 is pure 1-step TD (low variance, high bias). λ = 1 is Monte Carlo (high variance, low bias).

The key consequence for code: **GAE needs a value function, so you're training a second model at the same time.** That means two losses to balance.

## Slide: "Why pay the cost of PPO?"

| You get | You pay |
|---|---|
| **Lower variance** (per-token credit via GAE + critic) | **Four models in memory** (policy, value, reference, reward model) |
| **Token-level credit**: each token gets its own advantage | **Fragile value-function initialization** |
| **Sample reuse**: several gradient steps per batch | **More hyperparameters** to tune |

The historical reason PPO was the default: in the late 2010s and early 2020s it was "by far the most developed and understood RL algorithm," and **outside language models it was far better than REINFORCE**. Lambert: "so much about language models is getting the implementation right," and PPO was the algorithm people knew how to get right.

---

## Slide: "PPO training loop in code"

```text
for each batch:
  1. Sample prompts from dataset
  2. Generate completions with current policy π_θ
  3. Score with reward model → per-sequence rewards
  4. Compute ref model log-probs → per-token KL penalty
  5. Shape rewards: r_t = r_t - β * KL_t (per token)
  6. Compute returns via backward pass (GAE or MC)
  7. Compute advantages: A_t = returns_t - V(s_t)
  8. For k = 1 to K epochs on this batch:
     - Compute ratio = π_θ(a_t|s_t) / π_old(a_t|s_t)
     - Policy loss: clipped surrogate objective
     - Value loss: MSE on returns
     - total_loss = policy_loss + vf_coef * value_loss
     - Backward + optimizer step
  9. Sync π_old ← π_θ
```

This is the same loop as `../lecture-3-policy-gradients/04-ppo.md`, now numbered the way the code is organized. One subtlety Lambert highlights in steps 3–5: **the reward model gives one number per sequence, but the KL penalty is per token.** You have to combine two things of different shapes.

---

## Slide: "From scalar reward to per-token returns"

```
r_t = R(x, y) − β·KL_t    if t is the final token
r_t =         − β·KL_t    otherwise

KL_t = log π_θ(a_t | s_t) − log π_ref(a_t | s_t)
```

This is lecture 3 `02`'s shaped reward, as an array: shape (B, L), mostly small negative KL penalties, with the reward-model score added at each sequence's **last real token**.

### Lambert's live correction

He first said the reward gets "broadcast across every token," then checked mid-lecture and corrected himself:

| Setup | How the sequence reward reaches the tokens |
|---|---|
| **PPO with a value model (RLHF)** | Reward on the **final token only**, KL on every token. GAE then spreads credit backward. **This is what the slide shows.** |
| **No value model (GRPO, RLVR)** | Compute one advantage per sequence and **broadcast that advantage** to every token |

"Because the value model provides per-token credit assignment, this math works. If you are not using a value model, you don't do this and you broadcast the advantage."

---

## Slide: "What a PPO step uses"

Everything the loss needs has to be computed during **rollout** (generation) and stored:

| Tensor | Shape | What it is | Used by |
|---|---|---|---|
| Token IDs | (B, L) | Prompt + completion tokens | Everything |
| Completion mask | (B, L) | 1 = completion token, 0 = prompt/padding | Everything |
| **Old log-probs** | (B, L) | Policy's log-probs **at the start of this batch** (π_old) | The ratio ρ |
| **Old values** | (B, L) | Critic's predictions **at the start of this batch** | GAE, value-loss clipping |
| **Ref log-probs** | (B, L) | Reference model's log-probs (π_ref) | KL penalty |
| Rewards | (B,) or (B, L) | Reward-model scores (per sequence, or shaped per token) | Advantages |

Lambert repeats the warning from lecture 3: **π_old ≠ π_ref.** π_old is from the start of the current RL step. π_ref is from the start of the whole run. "The code is meshing all of these to get to one actual loss function."

## Slide: "Cached vs. recomputed" ⭐ the timing slide

| **Stored at rollout** (computed once, frozen for the batch) | **Recomputed at every training step** |
|---|---|
| Token IDs, completion mask | **New policy log-probs** log π_θ |
| Old log-probs log π_old | **Current value predictions** V_φ |
| Old value predictions V_φ_old | |
| Ref log-probs log π_ref | |
| Rewards | |

The ratio **ρ = π_θ / π_old uses one fresh number and one cached number.** That's what makes multiple epochs over the same batch possible: the cached half stays fixed while the fresh half moves as the model trains.

This is the code version of lecture 3's "what changes when" table. Everything on the left is computed **before** the inner loop. Everything on the right is computed **inside** it, every step.

---

## Slide: "Computing advantages with GAE" *(the scariest code in the lecture)*

```python
# rewards: (B,) terminal only (KL shaping omitted), values_old: (B, L) from rollout
B, L = completion_mask.shape
advantages = torch.zeros_like(values_old)
next_v = torch.zeros(B, device=values_old.device)
gae = torch.zeros(B, device=values_old.device)

last_idx = completion_mask.long().cumsum(-1).argmax(-1, keepdim=True)
done_mask = (torch.arange(L, device=values_old.device).unsqueeze(0) >= last_idx).float()
rewards_t = torch.zeros_like(values_old).scatter_(-1, index=last_idx, src=rewards)

for t in reversed(range(L)):
    not_done = 1.0 - done_mask[:, t]
    delta = rewards_t[:, t] + gamma * not_done * next_v - values_old[:, t]
    gae = delta + gamma * lam * not_done * gae
    advantages[:, t] = gae
    next_v = values_old[:, t]

advantages = advantages * completion_mask
targets = (advantages + values_old).detach()
advantages = advantages.detach()
```

*(Simplified: terminal reward only. With KL shaping, you'd add −β·KL_t to each token's reward before the loop.)*

You don't need every line. Lambert says to focus on the **reversed loop**: "GAE is doing credit assignment. It starts on the last token, and it works backwards."

### The three parts

**Part 1: setup (the `zeros` lines).** Empty arrays to fill in.

**Part 2: bookkeeping for sequences of different lengths.** A batch holds completions of different lengths, padded to L. The code has to know where each one **really ends**:
- `last_idx`: the position of each sequence's last real completion token. (Cumulative-sum the mask, and the argmax finds where the count stops growing.)
- `done_mask`: 1 from that position onward, meaning the episode is over and nothing after it counts.
- `rewards_t`: an array of zeros with the reward-model score **placed (`scatter_`) at each sequence's last real token**.

Getting `last_idx` wrong is one of the silent bugs on the debugging slide: using position `[:, -1]` (the last *slot*, which is often padding) instead of the last *generated* token.

**Part 3: the backward loop, which is GAE itself.** Walking from the last token to the first:

```
delta = reward at t + V(next position) − V(this position)     ← the TD residual from lecture 3
gae   = delta + λ × (gae from the position after)             ← accumulate, with decay
```

(With γ = 1 for LLMs. `not_done` zeroes things past the end of a sequence.)

It runs backward because each position's advantage **includes a decayed version of the advantage after it**. You have to compute the later ones first.

### A worked example *(my addition)*

One completion, 4 tokens, final reward 1. The critic's value predictions at each position: **[0.5, 0.5, 0.5, 0.95]**. The critic jumps confidently at position 4.

TD residuals (`delta`):

| Position | reward | V(next) | V(this) | delta |
|---|---|---|---|---|
| 4 (last) | 1 | 0 (episode over) | 0.95 | **+0.05** |
| 3 | 0 | 0.95 | 0.5 | **+0.45** ← the jump |
| 2 | 0 | 0.5 | 0.5 | 0 |
| 1 | 0 | 0.5 | 0.5 | 0 |

The advantages depend on λ:

| Position | **λ = 0** (just delta) | **λ = 0.95** (delta + 0.95 × next one's advantage) |
|---|---|---|
| 4 | 0.05 | 0.05 |
| 3 | **0.45** | 0.45 + 0.95×0.05 = **0.498** |
| 2 | 0 | 0 + 0.95×0.498 = **0.473** |
| 1 | 0 | 0 + 0.95×0.473 = **0.449** |

**This refines the lecture 3 picture.** With λ = 0, credit lands exactly on the token where the critic's prediction jumped. With the typical λ = 0.95, credit **leaks backward** to earlier tokens, shrinking by 5% per token. In a 4-token example that barely shrinks anything. In real completions hundreds of tokens long, it fades over roughly the last ~20 tokens before the jump (1/(1−0.95) = 20). So λ is effectively "how far back should credit for a good moment reach?"

### The last three lines

- `advantages * completion_mask`: zero out non-completion positions.
- `targets = (advantages + values_old).detach()`: the critic's **training targets** (next slide).
- `.detach()` on both: **freeze these numbers.** They're data for the loss, not something to backpropagate through (`00-code-map.md`).

Lambert: "it's not the most crucial thing, but it is good for learning about the core of RL."

---

## Slide: "PPO policy loss in code"

```python
# Compute probability ratio
ratio = torch.exp(new_per_token_logps - per_token_logps)  # (B, L), per_token_logps cached from rollout

# Clipped surrogate objective
eps = 0.2  # clip range
pg_losses1 = -advantages * ratio
pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - eps, 1.0 + eps)
pg_loss = torch.max(pg_losses1, pg_losses2)
```

### Line by line

**`ratio = exp(new − old)`.** ρ = π_new / π_old, but computed in log space: subtract the log-probs, then exponentiate. That's the same number (because log(a/b) = log a − log b), and it's numerically safer than computing two tiny probabilities and dividing.

**`pg_losses1 = −A × ρ`.** The unclipped surrogate. Lambert points out this is REINFORCE's loss with `seq_log_probs` replaced by `ratio`, "so you can take multiple steps on the data."

**`pg_losses2 = −A × clamp(ρ, 0.8, 1.2)`.** The same with ρ clipped to the band. `clamp` means "force into this range."

**`torch.max(…)`: max, not min.** Lambert calls this "a common interview question":
- The math (lecture 3) takes the **min** of two **objectives** (maximized), which picks the more cautious one.
- The code has **flipped the signs** (losses, minimized). The more cautious choice is now the **larger** loss.
- **min of objectives = max of negated objectives.** Same thing, sign-flipped.

His intuition: picking the smaller loss would let the model "take a gigantic gradient step." `max` "checks the more pessimistic gradient update."

---

## Slides: "Multiple epochs and minibatches" ⭐ *(new detail beyond lecture 3)*

PPO and GRPO can reuse each rollout batch for several gradient steps. **Clipping only does anything once π_θ has drifted from π_old**, and there are **two** ways that happens:

### 1. Minibatching

The rollout batch is often too big to fit on the GPUs, since a large batch is what you want for low variance. So you split it into **minibatches** and take a gradient step on each.

The consequence: **after the first minibatch's update, the model has already changed.** Minibatch 2 of the *same pass* sees ρ ≠ 1, so **clipping can activate even with K = 1 epoch.**

```
rollout batch (π_old log-probs saved)
  minibatch 1: ρ = 1 exactly (nothing changed yet)  → step → model changes
  minibatch 2: ρ ≠ 1 already                        → step → clipping can activate
  minibatch 3: ρ drifts further                     → step
  ...
```

### 2. Multiple epochs

Loop over the whole batch K times to learn more from each expensive rollout. Each pass sees a more-updated model, so the ratios drift further.
- **Typical K = 2–4.**
- **Past ~6, the policy is "too far off-policy"**: the batch no longer reflects the current model, even with clipping.

### When clipping never activates

> **With K = 1 and no minibatching, π_θ = π_old, every ratio is exactly 1, clipping never activates, and PPO reduces to vanilla policy gradient with GAE.**

Lambert says papers sometimes get confused about this: they describe "using a GRPO-like algorithm with clipping," but take one gradient step per batch with no minibatches, so the clipping never does anything.

Lecture 3 `05` noted that GRPO with one step per batch behaves like RLOO with z-scoring. This slide adds the precise condition: **one step per batch *and* no minibatching.**

Both PPO and GRPO use this same sample-reuse structure.

---

## Slide: "Value function targets"

The critic learns to predict **returns**: the total (shaped) reward from each token onward.

```
A_t = G_t − V_old(s_t)      (advantage = actual return − critic's prediction)

rearrange:
G_t = A_t + V_old(s_t)      (actual return = advantage + critic's prediction)
```

So **the target for the critic is advantage + old value**, which is the `targets` line from the GAE code:

```python
targets = (advantages + values_old).detach()
```

Why this is the right target: G_t (built from what actually happened plus GAE) is a **better estimate of the true return than the critic's current guess**. So you train the critic toward it with regression:

```
minimize (V_φ(s_t) − G_t)²
```

From the worked example above (λ = 0.95), targets = advantages + values:

| Position | advantage | V_old | **target** |
|---|---|---|---|
| 1 | 0.449 | 0.5 | **0.949** |
| 2 | 0.473 | 0.5 | **0.973** |
| 3 | 0.498 | 0.5 | **0.998** |
| 4 | 0.05 | 0.95 | **1.0** |

The completion succeeded (reward 1), so the critic is told: "you should have been more optimistic at positions 1–3." That's the "learns by observing" cycle from lecture 3 `04`, in code.

**`.detach()` is critical:** "targets are fixed from the rollout, not something we backpropagate through." Without it, the optimizer could lower the loss by **moving the target** instead of improving the prediction, and gradients would leak into places they shouldn't.

---

## Slide: "PPO critic loss in code"

The critic has **its own clipping**, which is "easy to overlook":

```python
# Current critic predictions
v_pred = value_net(completions)  # (B, L)

# old_values: critic predictions from rollout time (before any training updates, gradient detached)
# Clamp new predictions to stay within eps of the rollout values
v_clip = torch.clamp(v_pred, old_values - eps, old_values + eps)
vf_unclipped = 0.5 * (v_pred - targets) ** 2
vf_clipped   = 0.5 * (v_clip - targets) ** 2
vf_loss = torch.max(vf_unclipped, vf_clipped)
```

It's the same idea as the policy clip, applied to the critic: **don't let the critic's predictions move more than ε from its rollout-time predictions in one batch.**
- `vf_unclipped`: the normal regression (MSE) loss.
- `vf_clipped`: the loss if the prediction were forced to stay near the old value.
- `torch.max`: take the worse (more conservative) one. If the prediction is already close to the target, clipping doesn't interfere.

Lambert notes he "wasn't able to track down if it was in their original work," and that it's "one of these details that becomes default implementation over time."

---

## Slide: "PPO-RLHF: Combined objective"

```python
per_token_loss = pg_loss + vf_coef * vf_loss  # (B, L)

# Apply completion mask and aggregate
loss = ((per_token_loss * completion_mask).sum(dim=1) /
         completion_mask.sum(dim=1)).mean()
```

- **One combined loss** so that a single `backward()` updates both the policy and the critic.
- **`vf_coef`** (typically 0.5–1.0) balances "learn to act" against "learn to predict." It's another hyperparameter.
- **The KL isn't here.** In PPO it went into the **reward** (step 5), so it's already baked into the advantages. (GRPO does it differently, see `03`.)
- **The last line is one specific aggregation choice**: average within each sequence, then across sequences. `06` explains why this choice matters.

---

## Slide: "Advantage whitening"

```python
valid_adv = advantages[completion_mask.bool()]
advantages = ((advantages - valid_adv.mean()) /
              (valid_adv.std() + 1e-8)) * completion_mask
```

**Whitening** = z-score the advantages across the batch: subtract the mean, divide by the std, so they have mean 0 and spread 1. (It's GRPO's z-score idea, applied to **all valid tokens in the batch** rather than per prompt.)

**Why:** it keeps the **size** of the gradient stable from batch to batch. Without it, a batch where everything happened to get high (or low) advantages produces an outsized gradient. Lambert: "you want your loss size to be similar on each step," because "a bigger loss step can cause instability, and PPO can only control for that in so many ways."

- `advantages[completion_mask.bool()]`: only use real completion tokens for the statistics, not padding.
- `+ 1e-8`: a tiny number so you never divide by zero (`05`).

---

## Slide: "Value function initialization"

PPO was designed to learn **from scratch** in an environment. With LLMs you start from a strong pretrained model, and the critic needs a sensible starting point too:

| Option | Trade-off |
|---|---|
| **Initialize from the reward model** (the InstructGPT convention, and what Tülu 3 did) | Its predictions start near actual rewards, a good head start |
| **Initialize from the SFT model + a new random head** | Cheaper, but early training is unstable |

**The cold-start problem:** if the critic's early estimates are bad, GAE advantages are noisy, and "early training can be chaotic." That's lecture 3's "broken win-probability meter."

**The fix: value-function warmup.** Before updating the policy at all, take some training steps on **just the critic**, regressing on data with measured rewards, so it's stable before it starts steering the policy. Lambert did this in Tülu 3.

How a critic is built *(from the transcript)*: take a post-trained model and attach a **per-token output head** that predicts one number per position instead of next-token probabilities, then fine-tune.

---

## Slide: "PPO hyperparameters"

Illustrative ranges, not universal defaults:

| Hyperparameter | Typical range | What it controls |
|---|---|---|
| Clip ε | 0.1–0.2 | Trust-region width (lecture 3) |
| GAE λ | 0.95 | How far back credit reaches |
| Value coefficient | 0.5–1.0 | Weight of the critic loss vs. the policy loss |
| KL coefficient β | 0.01–0.1 | Strength of the leash to π_ref |
| Epochs per batch K | 2–6 | How much reuse ("off-policy budget") |
| Learning rate | **1×10⁻⁶ to 5×10⁻⁶** | **Much lower than SFT** |
| Batch size | 256–1024 prompts | Larger means lower variance |

Lambert's advice: "this is where you get death by complexity." **Find a stable recipe, then change one thing at a time.** Tuning everything at once "is literally hell."

---

## Slide: "Why batch size matters more in RL"

In supervised learning, gradient noise is moderate. In RL it's **orders of magnitude higher**, because gradients come from sampled rollouts, not labeled data. That's lecture 3's "it's an estimate, like a poll," in practice.

- **OpenAI's Dota 2:** the critical batch size was in the **millions** of transitions.
- **PPO isn't batch-size-invariant:** clipping ties batch size to the effective step size, so you can't just compensate with the learning rate.
- **PPO plateaus** are often caused by noisy loss estimates that only clear up with more samples.

**The takeaway:** larger batches reduce gradient variance roughly in proportion to 1/N. **"In RLHF, this is one of the cheapest ways to stabilize training, more effective than most hyperparameter tuning."** The simplest way to reduce variance is to sample more.

---

## Slides: "PPO debugging checklist"

### What to watch

| Metric | What healthy looks like | What it means if not |
|---|---|---|
| **Clip fraction** (% of tokens clipped) | Some, but not most | **0%:** clipping never activates (check for the K = 1, no-minibatch case or stale log-probs). **>50%:** the policy is changing too fast, and clipped tokens lose their signal |
| **Advantage stats** | Mean ≈ 0 (if whitened), stable std | Stuck near zero means no information |
| **Value loss** | Decreases, then stabilizes while rewards rise | Not decreasing means the critic isn't learning |
| **KL divergence** | Increases **gradually** | Exploding means "very bad things happening" |

Lambert on clipping: "if many prompts in your dataset are clipping, you're losing a ton of signal."

### Common silent bugs (training runs, but learns the wrong thing)

| Bug | What goes wrong |
|---|---|
| **Prompt tokens included in the policy loss** | Wastes gradient on tokens the model didn't choose (the mask is wrong) |
| **Reward scattered to the wrong position** (e.g. `[:, -1]` instead of the last *generated* token) | The reward lands on padding. GAE credit starts from the wrong place |
| **Stale log-probs used as "current"** | ρ stuck near 1, and clipping never activates. You're computing the ratio of old to old |
| **Zero-std GRPO groups** (all completions got the same reward) | Divide by zero, giving **NaN** advantages that poison training |
| **Value or policy loss computed outside the completion mask** | Training on padding |

Lambert: "there's a lot of ways that this could go wrong. I literally cannot make a lecture to just list all of them."

---

## Slide: "The four models in memory"

For a 7B-parameter model in fp16 (2 bytes per parameter, so roughly **2 GB per billion parameters**):

| Model | Size | Purpose |
|---|---|---|
| Policy | ~14 GB | Being trained |
| Value function | ~14 GB | Learned critic |
| Reference | ~14 GB | KL anchor (frozen) |
| Reward model | ~14 GB | Scoring (frozen) |

That's **~56 GB just for the weights**, before optimizer state, activations or gradients. *(My addition: the two trained models also need gradients and Adam's optimizer state, which roughly triples to quadruples their memory again.)* That's why PPO often needs the models split across GPUs (model parallelism) or parts moved to CPU memory (offloading).

There are tricks to cut this: e.g. **share one backbone** between policy and critic, with a separate value head. Lambert: "part of the reason GRPO is so popular is because it gets away from some of this."

*(Lambert also briefly repeats lecture 3's two layers of regularization, clipping per batch and KL per run, then says he'll remove that slide as redundant.)*

---

## The whole section in five lines

1. **Rollout computes and caches** token IDs, the mask, old log-probs, old values, ref log-probs and rewards. **Each training step recomputes** only new log-probs and current values. ρ = fresh ÷ cached.
2. **GAE in code** puts the reward on each sequence's last real token, then walks **backward**: delta = r + V(next) − V(this), accumulated with λ decay. λ = 0.95 means credit for a good moment leaks back over roughly ~20 tokens.
3. **Policy loss** = `torch.max(−A·ρ, −A·clamp(ρ))`. It's max because the signs are flipped. **Critic loss** = MSE toward `targets = advantages + old_values` (detached), with its own clipping. They're combined with `vf_coef`.
4. **Clipping activates from minibatching or extra epochs.** With K = 1 and no minibatches, ρ = 1 and PPO is plain policy gradient + GAE.
5. **In practice:** whiten advantages, initialize and warm up the critic, use a tiny learning rate and **big batches**, change one hyperparameter at a time, watch clip fraction, value loss and KL, and expect 4 models (~56 GB of weights at 7B).
