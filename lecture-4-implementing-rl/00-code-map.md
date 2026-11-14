# Code Map

A map for Lecture 4. It's the code companion to `../lecture-3-policy-gradients/00-training-loop-timeline.md`: the same training step, shown as **which code runs when**, **which tensors are cached vs. recomputed**, and **what each PyTorch operation means**.

Covers: `01`–`07` (the whole lecture).

---

## Shapes legend

Code comments like `# (B, L)` give a tensor's shape (its dimensions):

| Letter | Means | Example |
|---|---|---|
| **B** | Number of completions in the (mini)batch. Sometimes number of **prompts**, with B×G or B×K completions | 256 |
| **G** / **K** | Completions per prompt (GRPO says G, RLOO says K) | 4–16 |
| **L** | Sequence length in tokens (prompt + completion, padded to the same length) | 2048 |
| **V** | Vocabulary size (number of possible tokens) | ~100,000 |

| Shape | One number per… | Examples |
|---|---|---|
| `(B,)` | completion | rewards, `seq_log_probs`, GRPO/RLOO advantages, GSPO's ρ |
| `(B, L)` | token | `token_log_probs`, masks, ratios, KL, PPO advantages, values, `per_token_loss` |
| `(B, L, V)` | token × vocabulary entry | `logits` (the raw model output) |
| `(B, 1)` | completion, ready to **broadcast** across tokens | `advantages.unsqueeze(1)` |

---

## "Loss" in one line

**Loss** = the single number training minimizes. **SFT:** `−log π(human token)`, which measures how well the model predicts the right answer. **RL:** `−advantage × log π(own sampled token)`, the same thing weighted by advantage. It's built for its *gradient*, so its value isn't meaningful; watch the reward instead. Full explanation in `01`.

---

## The training step as code (GRPO, the common case)

```
═══ ROLLOUT PHASE: once per batch (on actor GPUs in async setups, 04) ═══════════════════════

1. sample prompts
2. generate G completions per prompt with the policy (vLLM)        → token IDs (B·G, L)
                                                                    → completion_mask (B·G, L)
   old_logps = compute_log_probs(policy, ...)                       → CACHED (B·G, L)    ← π_old
3. rewards = reward_model(...) or verifier(...)                     → CACHED (B·G,)
4. ref_logps = compute_log_probs(reference, ...)  [if using KL]     → CACHED (B·G, L)
5. advantages = z_score(rewards.view(-1, G)) ... .unsqueeze(1)      → CACHED (B·G, 1)  ← FIXED

═══ TRAINING PHASE: every step, each minibatch, each epoch (learner GPUs) ═══════════════════

6. new_logps = compute_log_probs(policy, ...)                       → RECOMPUTED (B·G, L)
   ratio = exp(new_logps − old_logps)                               → (B·G, L)
   pg_loss = max(−A·ratio, −A·clamp(ratio, 1−ε, 1+ε))               → (B·G, L)
   per_token_kl = new_logps − ref_logps  (or another KL estimator)  → (B·G, L)
   per_token_loss = pg_loss + β·per_token_kl                        → (B·G, L)
   loss = ((per_token_loss·mask).sum(1) / mask.sum(1)).mean()       → scalar  ← aggregation (06)
   loss.backward()
7. optimizer.step()                                                 ★ policy weights change ★

→ more minibatches/epochs on the same cached batch, or back to 1
```

On the `per_token_kl` line: `new − ref` is the simplest of three per-token KL estimators (k1, k2, k3), which are explained in `../lecture-10-regularization/`. On-policy distillation (lecture 7) uses the same line with **teacher** log-probs in place of `ref_logps`, as the advantage rather than a penalty.

## PPO's differences

```
ROLLOUT PHASE adds:
   old_values = value_net(...)                                      → CACHED (B, L)
   rewards_t: RM score scattered to last_idx, minus β·KL per token  → (B, L)   ← KL goes HERE
   advantages, targets = GAE backward loop (02)                     → CACHED (B, L), detached
   advantages = whiten(advantages)                                  → (B, L)

TRAINING PHASE adds:
   v_pred = value_net(...)                                          → RECOMPUTED (B, L)
   vf_loss = max((v_pred − targets)², (clamp(v_pred, old±ε) − targets)²)
   per_token_loss = pg_loss + vf_coef·vf_loss                       ← no KL term here
   optimizer.step()                                                 ★ policy AND critic change ★
```

---

## Cached vs. recomputed

| Tensor | Computed | Frozen for | Why |
|---|---|---|---|
| Token IDs, completion mask | rollout | the batch | the text doesn't change |
| **Old log-probs** (π_old) | rollout | the batch | the denominator of ρ |
| **Ref log-probs** (π_ref) | rollout | the batch (π_ref is frozen for the whole run anyway) | KL |
| Rewards | rollout | the batch | the reward model / verifier is frozen |
| **Advantages** (and PPO targets) | rollout | the batch, `.detach()`ed | fixed data for the loss |
| Old values (PPO) | rollout | the batch | GAE + value clipping |
| **New log-probs** | **every step** | – | the policy changes every step → the numerator of ρ |
| **Current values** (PPO) | **every step** | – | the critic changes every step |

**The rule:** anything in the denominator or "fixed data" side is cached. Anything that depends on the weights being trained is recomputed. ρ = fresh ÷ cached.

*(DPO, lecture 6, applies the same rule with no rollout phase: ref log-probs for every pair are cached **once for the whole run**, and only the policy's log-probs are recomputed each step. See the DPO cached-vs-recomputed table in `../lecture-6-dpo/07-dpo-implementation-details.md`.)*

## `compute_log_probs` is called on three models at three times

| Call | When | Becomes |
|---|---|---|
| `compute_log_probs(policy, …)` | rollout | `old_logps` (π_old) |
| `compute_log_probs(reference, …)` | rollout | `ref_logps` (π_ref) |
| `compute_log_probs(policy, …)` | every training step | `new_logps` (π_θ) |

In async setups (`04`), `old_logps` often come straight from the generation engine (vLLM), from weights that may be a step or two old.

---

## Where each lecture 3 concept lives in code

| Lecture 3 concept | Code | Notes file |
|---|---|---|
| ∇ log π ("which knobs") | `token_log_probs` + `loss.backward()` | `01` |
| Weighted supervised learning | `−(advantages * seq_log_probs).mean()` | `01` |
| Maximize → minimize | the leading **minus sign** | `01` |
| Batch baseline (REINFORCE) | `rewards.mean()` | `01` |
| Leave-one-out (RLOO) | `(row.sum(keepdim) − row) / (K−1)` | `01` |
| Group z-score (GRPO) | `view(-1, G)`, `.mean`, `.std`, `+1e-4` | `03` |
| **Advantage copied to every token** | **`.unsqueeze(1)` + broadcasting** | `03` |
| Shaped per-token reward (PPO) | `scatter_` at `last_idx`, `− β·KL` | `02` |
| TD residual + GAE | the `for t in reversed(range(L))` loop | `02` |
| Critic learns by observing | `targets = (advantages + values_old).detach()`, MSE | `02` |
| Ratio ρ | `exp(new_logps − old_logps)` | `02` |
| Clipping (off switch) | `torch.max(−A·ρ, −A·clamp(ρ))` | `02` |
| KL in reward vs. in loss | PPO: in `rewards_t`. GRPO: `+ beta * per_token_kl` | `03` |
| GSPO's per-sequence ratio | `exp((log_ratio·mask).sum(1) / mask.sum(1))` | `03` |
| CISPO's cap (stop-gradient) | `clamp(ρ).detach() * A * new_logps` | `03` |
| One gradient per batch | `.mean()` / `.sum()/mask.sum()` → scalar → `backward()` | `01`, `06` |

---

## PyTorch operations cheat sheet

| Operation | Plain meaning | Where it appears |
|---|---|---|
| `model(input_ids).logits` | Run the model and get raw scores for every vocab entry at every position, (B, L, V) | log-probs |
| `x[:, :-1]` / `x[:, 1:]` | Drop the last / first position along the sequence (**the autoregressive shift**) | log-probs |
| `.log_softmax(dim=-1)` | Raw scores → log-probabilities over the vocabulary, computed safely | log-probs |
| `.gather(dim=-1, index=…)` | From each position's ~100k log-probs, **pick the one for the actual token** | log-probs |
| `.unsqueeze(k)` / `.squeeze(k)` | Add / remove a size-1 dimension so shapes line up | everywhere |
| `x * mask` | Zero out positions where mask = 0 (prompt, padding) | everywhere |
| `.sum(dim=1)` | Sum along the sequence: one number per completion | aggregation |
| `.sum()` (no dim) | Sum **everything** into one number | per-token aggregation |
| `.mean()` | Average | everywhere |
| `.view(-1, G)` / `.reshape(-1)` | Reshape: flat list ↔ grid of (prompts × G). Needs **prompt-major order** | RLOO, GRPO |
| `keepdim=True` | Keep the summed dimension as size 1, so the result can broadcast back | RLOO, GRPO |
| `.repeat_interleave(G)` | Repeat each element G times: [a, b] → [a, a, a, a, b, b, b, b] | GRPO |
| **Broadcasting** | (B, 1) × (B, L): PyTorch auto-stretches the size-1 dimension | advantage to every token |
| `torch.exp(a − b)` | Ratio from log-probs (e^(log x − log y) = x/y), numerically safe | ρ |
| `torch.clamp(x, lo, hi)` | Force values into [lo, hi] | clipping |
| `torch.max(a, b)` / `torch.min(a, b)` | Elementwise larger / smaller of two tensors | PPO/GRPO clip, value clip |
| **`.detach()`** | **Treat as a fixed constant: no gradient flows through it** (stop-gradient) | advantages, targets, CISPO |
| `.scatter_(dim, index, src)` | Write values into specific positions (e.g. the reward at each sequence's last token) | PPO reward shaping |
| `cumsum` / `argmax` | Running total / position of the max. Used together to find each sequence's last real token | GAE bookkeeping |
| `.clamp_min(1)`, `+ 1e-8` | Division guards: never divide by 0 | everywhere you divide |
| `.to(torch.float32)` | Convert to full precision (safer math) | log-probs |
| `loss.backward()` | Autograd: compute the gradient of the loss for every weight | the end of every loss |

---

## Silent-bug map: where things go wrong

| Step | Bug | Symptom |
|---|---|---|
| 2 (mask) | Prompt or padding tokens left in `completion_mask` | Wasted gradient. Model trains on things it didn't choose |
| 2 (layout) | Completions not grouped by prompt before `view(-1, G)` | Baselines mix different prompts (wrong advantages, no crash) |
| 3 (reward) | Reward placed at `[:, -1]` instead of `last_idx` | Reward lands on padding. GAE starts from the wrong place |
| 5 (advantage) | All G rewards equal → std = 0, with no `+eps` | **NaN** advantages |
| 5 (advantage) | Advantages not `.detach()`ed | Gradients flow through the reward/critic computation |
| 6 (ratio) | `old_logps` accidentally recomputed with the current policy | ρ ≡ 1, clip fraction 0%, PPO "does nothing" |
| 6 (log-probs) | `softmax` then `log` | Tiny probabilities → 0 → −inf → **NaN** |
| 6 (aggregation) | Empty completion (immediate EOS), dividing by `mask.sum()` = 0 | **NaN** |
| 6 (PPO) | `targets` not detached | The critic "learns" by moving its own target |
| 6 (aggregation) | Unexamined normalization choice | Answer length drifts during training (`06`) |
| 2 → 6 (async) | `old_logps` from the inference engine (vLLM) don't match the trainer's for the same weights (low-precision LM head) | ρ ≠ 1 with no real change, so spurious clipping and instability. Fix: run the LM head in FP32 (lecture 5 `02`, MiniMax-M1) |
| 2 (multi-turn) | Tool-output or template-inserted tokens left in `completion_mask` | Model trains on text the environment wrote. In agentic rollouts the mask must be **striped**: 1 only on tokens the model sampled (lecture 11 `03`) |
| 2 → 6 (multi-turn) | Rollout re-rendered from text and re-tokenized instead of keeping the sampled token IDs | Tokens and saved `old_logps` no longer line up, so ρ is garbage with no crash. Keep the sampled IDs (lecture 11 `03`) |
