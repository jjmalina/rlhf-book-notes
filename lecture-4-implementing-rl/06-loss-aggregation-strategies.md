# Loss Aggregation Strategies

Notes on *Implementing RL Algorithms for LLMs | Post-Training Course, Lecture 4* (Nathan Lambert), 45:04–52:08. Based on `source/lecture-4-implementing-rl-slides.md` and `source/lecture-4-implementing-rl-transcript.md`. Follows on from `05-practical-engineering.md`. Also covers the "What to monitor" and "Regularization" forward-pointer slides, which come at the end of this stretch.

---

## Where this fits

Every algorithm produces a **per-token loss**: one number per token, per completion, shape (B, L). To call `backward()`, you need **one number**. The question here is how you average them.

It sounds like a trivial detail, but it isn't:

> **"Same algorithm, different aggregation → different training dynamics."** It's often visible in **how answer length changes during training**, especially in RLVR.

This became a big topic in the RLVR/reasoning era, when people saw that changing **one normalization line** made models write noticeably shorter or longer answers.

The setup, in lecture 3 terms: averaging decides **how much weight each token gets in the batch's single gradient**. Different averaging means different weights per token, which means a different gradient.

---

## Slide: "Bandit-style vs MDP-style RLHF"

Two ways to think about what one "action" is:

| | **Bandit-style** | **MDP-style** |
|---|---|---|
| The action is… | the **whole completion** | **each token** |
| Reward / advantage | One per completion, broadcast to every token | Per-token values / advantages |
| Default in | REINFORCE, RLOO, GRPO | PPO with GAE |

**"Bandit"** comes from the slot machine ("one-armed bandit"): you pull one lever (take one action), things play out, and you get one outcome. RLHF with a reward model or verifier looks like this: write a whole answer, get one score.

An **MDP** (Markov decision process, the formal setup from lecture 3) treats each step (token) as its own decision with its own credit.

**Practice is a mix:** "sequence-level rewards, but token-level log-prob gradients." Even GRPO computes its loss per token (per-token ratios, per-token clipping); it just gives every token the same advantage. And PPO-RLHF starts from a sequence-level score, then manufactures token-level credit with KL shaping and GAE.

That mix is why aggregation is ambiguous: should each **completion** count equally (bandit thinking) or each **token** (MDP thinking)?

---

## The three strategies

The notation: ℓ_{i,t} = the per-token loss for token t of completion i. |a_i| = the length of completion i. B = the number of completions.

### Strategy 1: Per-sequence normalization (each completion counts equally)

```
L = (1/B) Σ_i  (1/|a_i|) Σ_t ℓ_{i,t}
     average over completions of (average over that completion's tokens)
```

```python
loss = ((per_token_loss * completion_mask).sum(dim=1) /
         completion_mask.sum(dim=1)).mean()
```

- `.sum(dim=1) / completion_mask.sum(dim=1)`: each completion's **own average** over its tokens
- `.mean()`: then average those across completions

This is **standard in GRPO** (the `1/|a_i|` in its formula) and some PPO implementations. It's the line at the end of both `02`'s PPO loss and `03`'s GRPO loss.

**Effect:** each completion contributes equally, so **each token's weight is 1/(its completion's length)**:
- short completion (5 tokens): each token gets ∝ 1/5 = **0.20**
- long completion (10 tokens): each token gets ∝ 1/10 = **0.10**

**Tokens in short answers get pushed harder.** The slide: "this can bias the model away from lengthy responses."

### Strategy 2: Per-token normalization (each token counts equally)

```
L = (Σ_i Σ_t ℓ_{i,t}) / (Σ_i |a_i|)
     sum every token's loss in the batch ÷ total number of tokens in the batch
```

```python
loss = (per_token_loss * completion_mask).sum() / completion_mask.sum()
```

This comes from **DAPO** (a widely-cited 2025 RLVR paper). Note there's no `dim=1`: it sums over **everything** at once.

**Effect:** every token gets the **same weight**, however long its answer is. So **longer answers contribute more total gradient**, simply because they have more tokens. The slide: "can bias toward verbose completions."

Lambert on the motivation: so that "the tokens are handled equally [between long and] shorter sequences, to let the model understand which strategy is best on the individual merit." DAPO "was extremely popular for potentially making RL algorithms do this kind of inference-time scaling a bit better and learn to solve harder problems faster." Inference-time scaling here means reasoning longer to solve harder problems.

### Strategy 3: Fixed-length normalization (divide by a constant)

```
L = (1/B) Σ_i  (1/L_max) Σ_t ℓ_{i,t}
     like per-sequence, but divide every completion by the same max length
```

```python
loss = ((per_token_loss * completion_mask).sum(dim=1) /
         L_max).mean()
```

This comes from **Dr. GRPO**. It replaces "÷ this completion's length" with "÷ a fixed constant," e.g. the maximum generation length.

**Effect:** every token gets the same weight (like per-token), and longer answers contribute more total gradient (more tokens in the sum).

**Per-token vs. fixed-length:** Lambert looked this up mid-lecture. Both remove the length bias. The difference is just a **scale factor**: per-token divides by the batch's total token count, which changes from batch to batch, while fixed-length divides by a constant. A scale factor "can be absorbed into the learning rate." He describes fixed-length as a form of per-token that's "a bit numerically more stable."

---

## Slide: "Comparing aggregation strategies" (worked example)

Two completions with per-token losses. Each ends in one big-loss token (10):

```python
seq_1_losses = [1, 1, 1, 1, 10]                 # 5 tokens,  sum = 14, mean = 2.8
seq_2_losses = [1, 1, 1, 1, 1, 1, 1, 1, 1, 10]  # 10 tokens, sum = 19, mean = 1.9
```

| Strategy | Batch loss | Weight per token, short seq | Weight per token, long seq |
|---|:---:|:---:|:---:|
| **Per-sequence** | (2.8 + 1.9)/2 = **2.35** | **0.20** | **0.10** |
| **Per-token** | (14 + 19)/15 = **2.2** | 0.067 | 0.067 |
| **Fixed-length** (L_max = 10) | (1.4 + 1.9)/2 = **1.65** | 0.10 | 0.10 |

*(The slide's weights leave out the shared 1/B factor. Only the comparison between columns matters.)*

How to read it:
- **Per-sequence:** the short answer's tokens get **twice the weight** of the long answer's. Look at the "10" token: its effect is diluted by 5 tokens in seq 1 but by 10 tokens in seq 2.
- **Per-token and fixed-length:** every token has equal weight. The long sequence matters more overall, because it has more tokens.

---

## Why length bias happens *(my addition, connecting to lecture 3)*

In lecture 3, each completion's grad-log-prob vector was multiplied by its advantage. Aggregation adds **one more multiplier per token**:

```
token's push  =  advantage  ×  aggregation weight  ×  which-knobs
                               ^^^^^^^^^^^^^^^^^^
                               1/|a_i| under per-sequence; constant under per-token / fixed
```

Take two **correct** answers in RLVR with the **same advantage** (+1), one short and one long (Lambert's example). Under per-sequence normalization, every token of the short one gets a bigger push, so the model learns "be correct **and short**" more strongly than "be correct and long."

Aggregation isn't the only source of length drift. Human preference labels lean toward longer answers too (verbosity bias), so a reward model trained on them pays for length directly. See `../lecture-8-preference-data/04-open-questions-in-rlhf-data.md`, and the length-fixes table in `../lecture-9-over-optimization/` (`05`).

The sign of the advantage matters too. This is the core argument of the Dr. GRPO paper, which goes beyond the slides: for a **wrong** answer (negative advantage), per-sequence normalization **penalizes each token of a long wrong answer less** than a short wrong one. So being wrong at length is punished more gently, and wrong answers can drift **longer** over training. Per-sequence normalization leans toward short correct answers and long wrong ones, which is why people saw lengths change in ways they hadn't intended. Per-token and fixed-length normalization remove that 1/|a_i| factor.

**None of these is "correct" in general.** Each is a choice about what one unit of credit should be: one answer, or one token. It's worth knowing which one your codebase uses, because it shows up as length trends in training.

---

## Slide: "What to monitor during training"

Illustrative ranges, which vary by model size, task and algorithm. "W&B" = **Weights & Biases**, a common dashboard tool for training runs.

| Panel | Healthy | Unhealthy | What it means |
|---|---|---|---|
| `reward/mean` | Steady upward trend | Spikes, oscillation, plateau | Is it learning at all? |
| `kl/mean` | Gradual increase (0 → 2–5) | Explosion (>10) or flat at 0 | Drift from π_ref. Flat at 0 = not changing; exploding = running away (reward hacking?) |
| `loss/policy` | Decreasing | Diverging or NaN | NaN = the numerical issues in `05` |
| `metrics/clip_frac` | 5–30% | 0% or >50% | 0% = clipping never active (`02`); >50% = changing too fast, losing signal |
| `generation/length` | Stable or a slight increase | **Monotonic increase** | **A "length hack"**: the model found that longer = more reward (or the aggregation bias above) |
| Policy entropy | Slow decrease | **Crashes to 0** | See below |

**Policy entropy** *(term explained)*: how **spread out** the model's next-token probabilities are. High entropy means it considers many options; low means it's confident. A **slow decrease** is normal, since the model gets more decisive as it learns. A **crash to 0** means **mode collapse**: the model writes the same thing every time and has stopped exploring. With GRPO that's especially bad, because identical answers mean identical rewards, zero advantages, and **no learning signal at all** (lecture 3 `05`).

Also: **watch held-out eval scores, and actually read sample outputs** for coherence. Numbers can look fine while the text turns into nonsense.

*(Cross-lecture note: during over-optimization `reward/mean` looks **healthy**. It keeps climbing while true quality falls, so it can't warn you on its own. You need a held-out reward model or periodic evals to catch it: `../lecture-9-over-optimization/03-signatures-of-over-optimization-and-misalignment.md`. When and how evals run relative to the loop: `../lecture-12-evaluation/00-map.md`.)*

Lambert: GRPO-style training is "much simpler to [monitor] than PPO. The reward should be going up. Your KL distance should look reasonable. You should not have loss spikes. You don't want too much clipping. You don't want wildly unpredictable generation length, like collapse or exponential increase."

---

## Slide: "Forward pointer: Regularization & over-optimization"

"When RL runs go wrong, it is often due to RL latching onto spurious signals rather than the intended reward." These get their own future lecture (book chapters 14 & 15):

- **KL penalties:** forward vs. reverse KL, math and code
- **Goodhart's Law:** "when a measure becomes a target, it ceases to be a good measure," i.e. when the proxy reward diverges from true quality
- **Reward hacking:** sycophancy (telling people what they want to hear), verbosity, over-refusal
- **Implicit regularization:** papers like "SFT Memorizes, RL Generalizes" and "RL's Razor"

Lambert's RLVR example of reward hacking: with a verifier for instruction-following, a model might discover that "if I output no tokens, I always get reward." This happens in RLVR too, not only with learned reward models.

**Now covered:** over-optimization and Goodhart in `../lecture-9-over-optimization/` (`02`, with the signatures table in `03`); KL directions, estimators and implicit regularization (RL's Razor) in `../lecture-10-regularization/`.

*(The slides also contain a commented-out, unused slide on "Extended RL training practices": removing KL for long runs, asymmetric clipping (DAPO), progressive difficulty, keeping prompts where 20–80% of completions succeed, and dropping prompts where all completions are right or all wrong. It isn't presented, but it previews the reasoning lecture. The last two items follow directly from GRPO's "no signal when all rewards are equal.")*

---

## The whole section in four lines

1. **You have to turn a (B, L) grid of per-token losses into one number**, and how you average decides each token's weight in the gradient.
2. **Per-sequence** (GRPO default): each completion counts equally, so tokens get 1/length weight, which favors short answers (and gentler penalties for long wrong ones). **Per-token** (DAPO): each token counts equally, and long answers contribute more. **Fixed-length** (Dr. GRPO): ÷ a constant, which equals per-token up to a scale factor.
3. **Bandit vs. MDP framing** explains the ambiguity: rewards are per-sequence, but gradients are per-token.
4. **Monitor:** rising reward, gradual KL (0 → 2–5), clip fraction 5–30%, stable length (a monotonic rise = a length hack), slowly decreasing entropy (crash = mode collapse), and read the samples.
