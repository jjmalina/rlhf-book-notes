# DPO Implementation Details

Notes on *Direct Preference Optimization (DPO) and Friends | Post-Training Course, Lecture 6* (Nathan Lambert), 36:11–40:33. Based on `source/lecture-6-dpo-slides.md` and `source/lecture-6-dpo-transcript.md`. Follows on from `06-a-zoo-of-direct-alignment-algorithms.md`. See `00-map.md` for the DPO step next to the PPO/GRPO step.

---

## Where this fits

Three practical slides: **the code** (tiny), **the data** (where the real work is), and **offline vs. online** (the honest comparison with RL). Compare lecture 4, which needed a whole lecture for RL's code and infrastructure (`../lecture-4-implementing-rl/00-code-map.md`).

---

## Slide: "Implementation is genuinely simple"

"No generation during training, no separate reward model — the heart of the loss is a few lines":

```python
# log-prob gaps for policy and frozen reference
pi_logratios  = policy_chosen_logps - policy_rejected_logps
ref_logratios = reference_chosen_logps - reference_rejected_logps

# positive when policy shifts mass toward the chosen completion
logits = pi_logratios - ref_logratios
losses = -F.logsigmoid(beta * logits)
```

Lambert's walk-through: get the language model's outputs on the chosen and the rejected ("which is just your data"), take their log-probabilities, form the ratios, take a log-sigmoid weighted by β, "and then you would do `loss.backward()`. And this is mostly what DPO looks like and why it was so easy for people to get started."

| Line | Shape | Meaning |
|---|---|---|
| `policy_chosen_logps` etc. | `(B,)` | One number per response: the **sum of token log-probs** over the response tokens (prompt tokens masked out). B = number of pairs |
| `pi_logratios` | `(B,)` | How much more the **policy** prefers chosen over rejected |
| `ref_logratios` | `(B,)` | How much more the **reference** preferred chosen over rejected |
| `logits` | `(B,)` | The policy's preference *beyond* the reference's. ×β = the margin u from `04` |
| `-F.logsigmoid(beta * logits)` | `(B,)` | −log σ(u), the per-pair loss. Then `.mean()` over the batch |

*(My note: `F.logsigmoid(z)` computes log σ(z) in one numerically safe step. Computing `torch.log(torch.sigmoid(z))` can give log(0) = −∞ for very negative z.)*

**Tip on the slide:** "π_ref is frozen, so **precompute and cache its log-probs to cut peak memory ~50%**." Reference code: `code/direct_alignment/` in the RLHF book repo.

### The full training step, as code *(my addition, a sketch around the slide's four lines)*

Using `compute_log_probs` and the shapes from `../lecture-4-implementing-rl/00-code-map.md`:

```python
# ═══ ONCE, BEFORE TRAINING (optional caching pass) ══════════════════════════════════
with torch.no_grad():                                   # reference is frozen: no gradients
    for pair in dataset:                                # every (prompt, chosen, rejected)
        pair.ref_chosen_logp   = seq_logp(reference, pair.prompt, pair.chosen)    # 1 number
        pair.ref_rejected_logp = seq_logp(reference, pair.prompt, pair.rejected)  # 1 number
del reference                                           # free its memory for training

# ═══ EVERY TRAINING STEP ═══════════════════════════════════════════════════════════
for batch in loader(dataset):                           # step 1: a batch of PAIRS (no step 2, 3, 5)
    policy_chosen_logps   = seq_logp(policy, batch.prompt, batch.chosen)     # (B,) step 6
    policy_rejected_logps = seq_logp(policy, batch.prompt, batch.rejected)   # (B,)
    reference_chosen_logps   = batch.ref_chosen_logp                         # (B,) step 4: lookup
    reference_rejected_logps = batch.ref_rejected_logp                       # (B,)

    pi_logratios  = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps
    losses = -F.logsigmoid(beta * (pi_logratios - ref_logratios))
    loss = losses.mean()
    loss.backward()
    optimizer.step(); optimizer.zero_grad()             # step 7: ★ policy weights change ★

    # for logging (05): implicit rewards
    chosen_rewards   = beta * (policy_chosen_logps   - reference_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()
```

where `seq_logp` = lecture 4's `compute_log_probs`, then `(token_log_probs * response_mask).sum(-1)`. It reads the **given** text (teacher forcing, exactly like SFT). Nothing is sampled.

### Cached vs. recomputed in DPO *(my addition)*

| Tensor | Computed | Frozen for | Why |
|---|---|---|---|
| Token IDs of prompt, chosen, rejected | dataset creation | **the whole run** | the data never changes |
| **Reference log-probs** (chosen, rejected) | **once, before training** | **the whole run** | π_ref is frozen *and* the text is fixed |
| **Policy log-probs** (chosen, rejected) | **every step** | – | the policy changes every step |
| Implicit rewards, margin, loss | every step | – | depend on the policy log-probs |

Compare lecture 4's RL table: there, reference log-probs are cached only **per batch**, because every batch is new text. DPO can cache for the **whole run**. There's also **no π_old**, because no text comes from the policy.

**Why ~50% memory:** without caching, the reference model's full weights sit in GPU memory alongside the policy's for the whole run, doubling the model memory. With caching, it's loaded once, used for one pass, and freed. *(My note: the 50% is roughly about weights. The policy also needs gradients and optimizer state, so the real saving depends on the setup.)*

**Cost per step:** two teacher-forced forward passes (chosen, rejected) and one backward, roughly **2× an SFT step**. There's no generation, which lecture 4 `04` says dominates RL's cost.

---

## Slide: "DAAs work with synthetic preference data"

"These algorithms need *feedback* data, not necessarily *human* feedback data — AI feedback works just as well."

### UltraFeedback: the dataset that lit the fuse

- **The first prominent *open* synthetic preference dataset**, which "helped kickstart the DPO era."
- How it was made: prompts → completions from **a variety of open models** of the time → a **GPT judge** rates them on **multiple aspects** → pairwise rankings ("actually a pool of potential chosen and rejected pairs").
- Lambert: it "was like the match that made DPO methods take off with synthetic data." "Without the kind of rise of synthetic judgment data and... UltraFeedback, DPO may not have taken off as soon."
- Tülu 3, SmolLM 2 and others followed with larger synthetic-feedback recipes. SmolLM's team did a lot of synthetic DPO data "because it had a high return on investment in terms of engineering time."

*(Correction: Lambert says UltraFeedback was "about 10,000 prompts." As I recall, the UltraFeedback paper reports about **64k prompts** (each with 4 completions, ~256k in total), and the commonly used binarized version has ~60k pairs. His point, that Tülu 3 scaled this kind of recipe up a lot further, still stands. The garbled transcript number for Tülu 3's scale is left out.)*

### On-policy data (lesson from Tülu 3)

"When trying to push the limits of DPO, it's really important to have **completions from your SFT model** inside the pairwise data."

```
instruction data ──SFT──► SFT model ──generate on your prompts──► completions (+ other models' completions)
                                                                            │
                                                     LLM judge scores them, forms chosen/rejected pairs
                                                                            │
                                                                            ▼
                                                   DPO on those pairs, starting from the SFT model
```

Lambert's guess at why it helps: "it could be related to these kind of partition function ideas where you need to make sure you're optimizing the right tokens."

*(My note on timing: "on-policy" here is **not** the same as in RL. In RL the data comes from the policy *at the current step*, regenerated every batch. In DPO it's generated **once, before training, by the starting (SFT) model**, then frozen. It's on-policy for the start of the run only. That's why Lambert later calls a variant "semi-online" (`08`). It probably helps because DPO's push-down/pull-up acts on text the model actually tends to write, instead of text it would never produce anyway.)*

### Delta learning (Geng et al., used in Olmo 3)

The **gap** between chosen and rejected matters more than which models produced them. Example: **Qwen3-32B wrote the chosen, Qwen3-0.6B wrote the rejected.** Neither has to be a great answer. There just has to be a clear quality difference.

Lambert: "there needs to be meaningful signal in your chosen response being better than your rejected response. And if all the answers are too similar, the models don't learn as well." He calls why DPO works so well "still underexplored when everybody is looking at RL."

*(My note: this echoes GRPO's need for disagreement within a group (`../lecture-5-reasoning-models/03-common-implementation-patterns.md`): no contrast, no signal.)*

Connects to: `../lecture-7-synthetic-data/` (Lambert pulls the synthetic data chapter forward to lecture 7) and `../lecture-8-preference-data/`.

---

## Slide: "DPO vs. RL: Offline vs. online"

| | **DPO and other DAAs (offline)** | **PPO / policy gradient (online)** |
|---|---|---|
| Data | A **fixed dataset** collected ahead of time. Sometimes "fixed off-the-shelf datasets on Hugging Face" | **Fresh completions** generated during training, scored by a reward model |
| Strengths | **Simpler, more stable, fast to iterate on data.** "Way fewer moving parts" | **Can explore new regions** → often **higher peak performance** |
| Weaknesses | Limited by the dataset's **coverage**: a slightly lower ceiling | More compute, more moving parts (**four models in memory**), more engineering |

*(My note: "coverage" means DPO can only ever learn about responses that appear in the pairs. If the best answer to a prompt looks like nothing in the dataset, DPO can't discover it. Online RL samples new answers and can find it, which is lecture 5's "discover vs. elicit" debate in another form.)*

The 2024 comparison papers (Ivison et al., *Unpacking DPO and PPO*; Xu et al.; Tajwar et al.): "most of them found that RL was a bit better, which I think is a true conclusion."

Connects to: `../lecture-3-policy-gradients/04-ppo.md` (the models PPO needs), `../lecture-4-implementing-rl/04-training-infrastructure.md` (why generation is the bottleneck).

---

## The whole section in four lines

1. **The code is four lines** on per-response summed log-probs: policy log-ratio − reference log-ratio, × β, −logsigmoid, mean, backward. No generation, no reward model, no π_old.
2. **Cache the reference log-probs once before training** (the text and π_ref are both frozen), then free the reference: about half the model memory. Each step costs about 2× an SFT step.
3. **The data is what matters:** synthetic AI-feedback pairs (UltraFeedback kicked it off), including completions from your own SFT model ("on-policy," generated once), with a clear quality gap between chosen and rejected (delta learning).
4. **Offline vs. online:** DPO is simpler, stabler and quicker to iterate. Online RL explores beyond the dataset's coverage and usually peaks a bit higher, at much higher cost.
