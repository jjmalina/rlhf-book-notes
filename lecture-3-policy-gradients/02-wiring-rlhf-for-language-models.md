# Wiring RLHF for Language Models

Notes on *Understanding Policy Gradient Algorithms for RL on LLMs | Post-Training Course Lecture 3* (Nathan Lambert), 25:38–28:38. A slide-by-slide breakdown, based on `source/lecture-3-policy-gradients-slides.md` and `source/lecture-3-policy-gradients-transcript.md`. Follows on from `01-policy-gradients-core-intuitions-and-maths.md`.

---

## Where this fits

The previous section gave you the core recipe:

> **Policy gradient = supervised learning on the model's own outputs, with each example weighted by its advantage.**

That recipe doesn't care where the reward comes from. Lambert says this directly: "you can understand RL just as if you were given a reward." This short section covers the setup the recipe plugs into for real LLM training: which models are involved, and one safety mechanism (the **KL penalty**) that changes the reward before it reaches the recipe.

The section answers three questions:
1. What models are running during RLHF?
2. How do you stop the model from "cheating" the reward model?
3. What are the two different "old" models you'll see in equations, and how are they different?

---

## Slide: "The RLHF training loop"

This slide revisits the pipeline diagram from the start of the lecture. Before getting into specific algorithms, it's worth naming the models that are running:

| Model | Role | Trained? |
|---|---|---|
| **Policy** π_θ | The LLM being trained. Generates the completions. | ✅ Yes, this is the only one whose weights we update |
| **Reward model** | Scores each completion (the 0.9 for "The answer is 4"). | ❌ Frozen (trained earlier, in the reward-model lecture) |
| **Reference model** π_ref | A frozen copy of the policy from *before* RL started. | ❌ Frozen |
| *(PPO only, later)* **Value model** | Predicts expected reward, used as the baseline. | ✅ Yes, trained alongside |

The first three are the standard setup. In RLVR the reward model is replaced by a verifier, which is a program such as "check the math answer", not a neural net.

**Takeaway:** RLHF isn't one model learning on its own. It's one model being trained while 1–2 frozen models judge and constrain it. That's a big part of why RLHF is expensive: several LLM-sized models sit in GPU memory at once.

---

## Slide: "The reference model"

The **reference model** is a frozen snapshot of the policy at the very start of RL, usually the SFT checkpoint from the instruction-tuning lecture. It never changes.

**Why keep it?** Because the reward model isn't a perfect judge. It's a neural net trained on a limited set of human preferences, and it has blind spots. The policy is being optimized *only* to raise the reward model's score, so given enough steps it will find and exploit those blind spots. This is called **reward hacking** or **over-optimization**. Examples:

- The reward model slightly prefers longer answers → the policy learns to ramble endlessly.
- The reward model likes confident language → the policy states wrong things confidently.
- Lambert's example: "I've seen many cases where the model just starts generating gibberish" that still scores well.

The reference model is an **anchor**. It represents "a sensible model that writes normal text." The policy is allowed to improve, but it gets penalized for drifting too far from that anchor.

An analogy from your supervised-learning background: this plays the role of **regularization** (like an L2 penalty that keeps weights near zero). Here, though, it keeps the model's *behavior* near the starting model's behavior instead of keeping the weights small.

---

## Slide: "KL divergence as regularization"

KL divergence is a standard way to measure how different two probability distributions are. You don't need the general definition here. The slide gives the per-token version that actually gets computed:

```
KL_t = log π_θ(token_t) − log π_ref(token_t)
```

In plain words: **for the token the policy picked, how much more likely does the current model find it than the original model did?**

### How it's computed (the three steps on the slide)

1. **Generate** a completion from the current policy (which you already do for the reward).
2. **Compute the log-probs** of those exact tokens under *both* models: the current policy and the frozen reference. This is one extra forward pass through the reference model, with no sampling and no gradients.
3. **Subtract.** That gives one KL number per token.

### When this happens (timing)

- **Before the gradient, not after.** The KL penalty is part of the reward, and the reward has to be known before the gradient can be computed. "Current policy" means the policy that just generated this batch, before this step's update. (See `00-training-loop-timeline.md`.)
- **Measured against the original model.** The reference is frozen at the start of the run. At step 500 the policy has absorbed 499 updates, so the KL measures *total drift since training started*, not how much the last update changed things.
- **Step 1:** policy = reference, so every token's KL is 0. As updates accumulate, KL can grow and the leash tightens.
- **"Forward pass with no sampling"** means you feed the reference the text the policy already wrote and read off the probability it would have given each exact token. One pass covers every position. It works like scoring a known label in supervised learning. "No gradients" means you never call `backward()` through the frozen reference.

### A worked example

Say for the prompt "What's 2+2?", the token "4":

| | Probability of "4" |
|---|---|
| Reference model (start of RL) | 0.30 |
| Current policy (after some training) | 0.60 |

```
KL = log(0.60) − log(0.30) = log(0.60 / 0.30) = log(2) ≈ 0.69
```

The current model is now twice as confident in "4" as the original model was. That counts as some drift, which is fine here because "4" is correct, but it still costs a little.

If the two models agree (both say 0.30), then KL = log(1) = **0**, and there's no penalty.

### How the penalty enters training: "shaped reward"

The KL is folded into the reward *before* computing advantages:

```
intermediate tokens:  reward_t = −β · KL_t
final token:          reward_T = R(completion) − β · KL_T
```

- **β (beta)** is a dial that sets how strong the leash is, for example 0.05 or 0.1. Bigger β means the model stays closer to the reference.
- Every token pays a small drift penalty, and the final token also receives the reward model's score.

Since we don't discount (γ = 1), for a whole completion this works out to:

```
total reward = reward model score − β × (sum of per-token KL)
```

**Example:** "The answer is 4" gets reward-model score 0.9. Say its tokens' KL values sum to 2.0, and β = 0.1:

```
total reward = 0.9 − 0.1 × 2.0 = 0.7
```

That 0.7 (not 0.9) is the reward that gets compared against the baseline to compute the advantage. **The policy-gradient recipe from the previous section is unchanged.** Only the reward it's fed is different.

This makes the trade-off explicit: the model can earn more reward-model score by changing its behavior, but each change it makes costs some KL penalty. It only drifts when the gain is worth the cost.

*(Detail you can skip: an individual token's KL can be negative, when the policy finds a token less likely than the reference did. Averaged over many samples, the true KL is always ≥ 0. This per-token number is a sample estimate of the true KL, the same "estimate from samples" idea as the policy gradient.)*

Lambert notes that this gets covered in more depth in the later lectures on over-optimization and regularization.

---

## Slide: "KL in RLHF vs. RLVR"

How much you need the leash depends on how trustworthy the reward is:

| | RLHF | RLVR |
|---|---|---|
| **Reward source** | Reward model (a learned, imperfect *proxy* for human preference) | Verifier (e.g. "is the math answer correct?"), which is ground truth |
| **Can it be exploited?** | Yes, it has blind spots | Much harder, since a wrong answer is just wrong |
| **KL penalty** | Critical | Often reduced or removed entirely |
| **Reference model** | Kept | Sometimes dropped altogether (e.g. DeepSeek R1), which saves GPU memory |

**The trend:** as rewards become more reliable, you need less regularization. Lambert adds a caveat: with very, very long RL runs, the KL penalty may become useful again, so this could change.

---

## Slide: "Two 'old' policies — don't confuse them"

Equations later in the lecture (especially PPO) refer to two different "old" versions of the model. They look similar and are often confused, but they do different jobs.

| | π_old ("old policy") | π_ref ("reference policy") |
|---|---|---|
| **What it is** | The policy as it was when the current batch was generated | The policy at the start of the *entire* RL run (SFT checkpoint) |
| **Scope** | **Batch-level** | **Run-level** |
| **How often it changes** | Every batch (it's just "the model a few steps ago") | Never. Frozen. |
| **Used for** | The importance-sampling ratio ρ (PPO, later) | The KL penalty (this section) |
| **If you drop it** | You can only take 1 gradient step per batch | Risk of reward hacking |

### An intuitive way to tell them apart

- **π_ref is the starting line.** "How far have we drifted since training began?" It guards against long-term drift and reward hacking.
- **π_old is the snapshot that generated this batch.** "How much has the model changed since it wrote these samples?"

### Why π_old exists *(preview of PPO, covered later)*

In the previous notes we saw that PPO takes **several** gradient steps on one batch, to get more value from each expensive round of generation. The catch is that after the first step the model has changed, so the batch no longer reflects what the *current* model would write. PPO handles this by comparing the current model's probabilities to π_old's probabilities on the same tokens. That comparison is the ratio ρ, and PPO uses it to correct for the mismatch and to cap how far each step can go.

If you only take one gradient step per batch (as in basic REINFORCE), then π_old and the current model are identical at the moment of the update, and you don't need π_old.

**Implementation note from the slide:** in many codebases, π_old isn't a separate copy of the model. "Old log-probs" are just the log-probs saved when the batch was generated, before the model was updated.

---

## Updated training loop

This is the loop from the previous notes, with this section's additions marked ★:

```
policy    = copy of SFT model     ← trained
reference = copy of SFT model     ← ★ frozen forever
reward_model                      ← frozen

for step in range(many_steps):
    generate N completions per prompt with the policy        (save their log-probs ← ★ "old log-probs")
    score each completion with reward_model                  → R
    ★ compute log-probs of the same tokens under reference
    ★ KL_t = policy_logprob_t − reference_logprob_t          (per token)
    ★ reward = R − β × sum(KL_t)                             (shaped reward)
    advantage = reward − baseline (e.g. group mean)
    loss = −(advantage × sum of token log-probs).mean()      ← same weighted-supervised-learning recipe
    loss.backward()
    optimizer.step()
```

---

## The whole section in four lines

1. **Several models are running:** the policy (trained), the reward model (frozen judge), and the reference model (frozen anchor). PPO adds a value model.
2. **The reward model can be cheated**, so the policy is penalized for drifting from the reference model. The KL penalty is per-token log-prob(current) − log-prob(reference), scaled by β.
3. **The penalty is subtracted from the reward** before advantages are computed. The policy-gradient recipe itself doesn't change.
4. **π_ref ≠ π_old.** π_ref is the frozen start of the whole run (used for KL). π_old is the snapshot that generated the current batch (used for PPO's ratio).
