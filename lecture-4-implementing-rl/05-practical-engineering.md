# Practical Engineering

Notes on *Implementing RL Algorithms for LLMs | Post-Training Course, Lecture 4* (Nathan Lambert), 42:34–45:04. Based on `source/lecture-4-implementing-rl-slides.md` and `source/lecture-4-implementing-rl-transcript.md`. Follows on from `04-training-infrastructure.md`.

---

## Where this fits

Throughout `01`–`03`, small details showed up in the code: `log_softmax`, masks, `.detach()`, `+ 1e-4`. This short section collects them into one checklist of **places where RL training silently goes wrong or blows up numerically**.

Lambert: "it's easy for me to show you correct code, but these are the things that people battled through."

"Blows up" usually means **NaN** (not a number) or **inf** (infinity). Once a single NaN gets into the loss, it spreads through the gradient into every weight, and the model is ruined.

---

## Slide: "Key numerical considerations"

### 1. Log-space arithmetic: `log_softmax`, never `softmax` then `log`

**Rule:** always compute log-probs directly with `log_softmax` (+ `gather`). Never compute probabilities with `softmax` and take `log` afterwards.

**Why:** computers store numbers with limited precision. A token the model considers very unlikely might have a probability like 10⁻⁵⁰.
- `softmax` produces that tiny number, which is too small to store, so it gets **rounded to exactly 0**.
- `log(0)` = **−infinity**. Any arithmetic with it gives NaN.
- `log_softmax` computes the log directly (≈ −115 for that token) without ever forming the tiny number. No problem.

The slide's phrasing: "softmax squashes small probabilities to tiny floats, then log amplifies the precision loss." Lambert: "the ordering here in the implementation is very intentional."

The same logic applies to the ratio in `02`: compute `exp(new_logp − old_logp)`, **subtracting in log space first**, rather than dividing two probabilities.

### 2. Masking padding tokens

`completion_mask` must be **1 only for real completion tokens**. It must exclude:
- **prompt tokens** (the model didn't choose them)
- **padding** after the end (empty filler to make sequences the same length)
- **the EOS token itself**, or include it, as long as you do so **consistently** (next point)

Multiply losses by the mask **before** aggregating. Lambert: "masking and padding tokens is really hard... it can be such a nightmare." His examples: "your reward model is assigning reward at the wrong EOS token, or your EOS token ID is wrong and your reward model no longer works."

### 3. Stop-token (EOS) handling

**EOS** ("end of sequence") is the special token the model writes to say "I'm done." Decide whether it's part of the loss or not, and **be consistent** everywhere: the mask, where the reward is placed (`last_idx` in `02`), the length counts. "Mishandling is a common silent error."

### 4. Sequence-length handling

Completions in a batch have different lengths, so how you average over tokens needs care. That's the whole next section, `06-loss-aggregation-strategies.md`.

### 5. Detaching baselines and targets

Use `.detach()` on anything that should be **treated as fixed data**, not something the optimizer can change: advantages, value targets, CISPO's clipped ratio.

Lambert: "there's a lot of places, especially in PPO, where if you don't detach gradients on an intermediate value, you're gonna backpropagate through the wrong model, or your reward model, your target value, or so on, which is gonna manipulate the dynamics wildly."

An example of what goes wrong: if `targets` in PPO's value loss isn't detached, the optimizer can reduce the loss by **moving the target toward the prediction** instead of improving the prediction. The loss goes down while nothing useful is learned.

### 6. Division guards

Anywhere you divide by `mask.sum()`, a sequence length or a standard deviation, make sure it can't be zero:
- `.clamp_min(1)` forces the denominator to be at least 1
- `+ eps` adds a tiny number (`1e-8`, `1e-4`)

**The classic case:** the model writes EOS as its **very first token**, an empty completion. Its mask sums to 0, and `loss / 0` gives NaN. Another one from `03`: a GRPO group where every completion got the same reward, so std = 0.

---

## Lambert's aside on coding agents

> "Maybe with coding agents, this slide doesn't matter. Who knows? The coding agents are remarkably good at RL. They are not that good at distributed systems."

His view is that AI coding assistants handle **loss-function code** like this well, but the **infrastructure** from `04` is still hard for them.

---

## Slide: "Beyond the loss formula"

The bridge to the next section:

> Every algorithm computes the same core gradient, ∇ log π × A. But **how you aggregate per-token losses into a single number changes training dynamics more than you'd expect.**

From the transcript: "it's almost always a per-token loss that we sum in order to take gradient updates. There are a lot of different ways that you aggregate the per-token loss... These aggregation methods change the shape of the gradient, and they change how each token will contribute."

---

## Checklist

| ✅ Do | Why |
|---|---|
| `log_softmax` + `gather`; ratios as `exp(Δlog)` | Avoid underflow → −inf → NaN |
| Mask = 1 only on real completion tokens | Don't train on prompt or padding |
| Treat EOS one consistent way | A common silent error |
| Normalize over variable lengths deliberately | It changes learning dynamics (`06`) |
| `.detach()` advantages, targets and baselines | Don't backprop through things that should be fixed |
| `clamp_min(1)` / `+ eps` on every denominator | Empty completions and zero-std groups → NaN |
