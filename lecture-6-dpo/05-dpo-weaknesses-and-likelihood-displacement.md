# DPO Weaknesses and Likelihood Displacement

Notes on *Direct Preference Optimization (DPO) and Friends | Post-Training Course, Lecture 6* (Nathan Lambert), 30:16–33:30. Based on `source/lecture-6-dpo-slides.md` and `source/lecture-6-dpo-transcript.md`. Follows on from `04-derivation-the-loss-and-gradient.md`.

---

## Where this fits

The derivation is done. The rest of the lecture is practice: "what DPO is doing today, what else you should know about it, and what the implementation looks like." This chapter covers two weak spots that come straight out of the loss's shape:

1. **Likelihood displacement:** the loss only sees the *margin*, so the chosen answer can get *less* likely.
2. **The static KL:** β fixes how far DPO goes, once, for the whole run.

Both motivate the variants in `06`.

---

## Slide: "A subtle risk: The chosen probability can fall"

**The core point:** the DPO loss only cares about the **margin** between the chosen and rejected log-ratios, "not their absolute values." Lambert: "It is not looking at where the chosen rewards are in space... there's not a clear incentive to just make the one reward higher."

So the model can lower the loss by pushing the **rejected** probability down **faster** than the chosen one, **while the chosen probability also falls.** This is called **likelihood displacement** (Razin et al., 2024; Ren et al., 2024).

### The numeric example, continued *(my addition)*

The same setup as `04`: β = 0.1, log π_ref(y_c) = −50, log π_ref(y_r) = −48.

| Scenario | log π_θ(y_c) | log π_θ(y_r) | r_c | r_r | margin | **loss** |
|---|---|---|---|---|---|---|
| A. Start | −50 | −48 | 0.0 | 0.0 | 0.0 | 0.693 |
| **D. Displacement** | **−53** (down!) | −60 (way down) | **−0.3** | −1.2 | +0.9 | **0.341** |

The loss **halved**, and the model "learned the preference." But the chosen answer is now e^(−3) ≈ **5%** as likely as it was under the reference. The model got better at the comparison by making *both* answers unlikely, the rejected one much more so.

**Where did that probability go?** A model's probabilities over all possible responses must add up to 1. If the chosen and rejected answers both lose probability, it moves to **other responses, ones that aren't in the dataset at all**. Nothing in the loss says where it should go.

That's what Lambert means by the effect being **"mediated through the partition function Z(x)"**: Z is the normalizer over *every* possible response, and "DPO can move probability space around in the partition function domain." *(My note: my reading is that the margin-only loss leaves the absolute level free, and Z, the "all other responses" pool, is where the freed probability goes. The slide doesn't spell out more than that.)*

### Why people worry about it

- It's "posited to push probability toward unaddressed, **off-distribution** behaviours." Lambert: "you don't know the side effects when it's out of distribution, because the optimization itself really does have this kind of open-ended feature of potentially moving out-of-distribution tokens."
- So a line of papers studied "making sure that DPO is actually making the correct reward go up, so the reward to be positive, and not just the margin."

### Fixes mentioned

- **Add an SFT term on the chosen response.** A plain −log π(y_c) term pushes the chosen answer up in absolute terms, not just relative to rejected. (ORPO in `06` builds this in.)
- **Cal-DPO** (Xiao et al., 2024) and **AlphaPO** (Gupta et al., 2025): loss modifications aimed at this. The slide just names them.

### Slide image and the real run

*(Neither image is in the sources. Reconstructed from the slide caption and transcript.)*

**The sketch:** chosen and rejected log-probs over training. Both go down, rejected faster, so the gap between them widens.

```
log-prob
   │ ───────────╲___________________  chosen   (drifts down)
   │             ╲
   │              ╲_______
   │                      ╲________  rejected (falls faster)
   │        ← margin keeps widening →
   └──────────────────────────────────── training steps
```

**A real run:** the RLHF book's Olmo 1B DPO run (a public W&B link on the slide). Lambert's description: "rejected rewards going down and chosen rewards are kind of flat and the margin is increasing." He admits it "isn't the super most indicative" example, and that "in a lot of work... chosen actually goes down."

*(My note: the "rewards" on these dashboards are the **implicit rewards**, β·(policy log-prob − reference log-prob), logged per response. They're cheap to compute because both log-probs are already computed for the loss. "Chosen reward flat" means the chosen answer's probability is about where the reference had it. A chosen reward below 0 means it's less likely than under the reference, which is displacement.)*

**What to watch in a DPO run** *(my addition, from the above)*:

| Logged value | Healthy | Warning sign |
|---|---|---|
| loss | starts at 0.693, goes down | – |
| margin (rewards/chosen − rewards/rejected) | goes up | – |
| **rewards/chosen** | flat or rising | **falling steadily** (likelihood displacement) |
| rewards/rejected | falling | – |
| accuracy (fraction of pairs with margin > 0) | rises above 50% | – |

---

## Slide: "The β parameter and the static KL"

β "sets the strength of the KL constraint relative to reward maximization," as in the RLHF objective it came from (`02`):

| β | Effect |
|---|---|
| **Large β** | Policy stays close to π_ref. "It barely moves" |
| **Small β** | Policy is free to deviate. It can **over-optimize** |

*(My note: you can see this in `02`'s toy example. β = 10 left π* almost equal to π_ref. β = 1 moved half the probability to the best answer.)*

**The crucial difference from RL: DPO's KL is static.** DPO "steps directly to the *optimal* solution implied by the dataset and the chosen β." The β you pick at the start defines one fixed target (`02`'s π*), and training heads straight for it.

Online RL (lectures 3–4) handles the KL differently:

| | Online RL (PPO/GRPO) | DPO |
|---|---|---|
| How KL enters | A per-sample penalty, computed on freshly generated text each batch (step 4–5 or 6) | Baked into the loss through β, via the derivation |
| What it controls | "How much the model can learn from every batch" | **The final distance from π_ref**, for the whole run |
| Can it change during the run? | Yes: some runs use **dynamic KL controllers** that raise or lower β as the KL drifts | Not really: "some of these sorts of things aren't really suited or possible in DPO frameworks" |

Lambert calls it "a subtle tuning point that you have to get right." DPO handles the term "much more directly," and it interacts with things like the learning-rate schedule.

*(My note on timing: in RL, the KL is measured as you go, on text the current policy just wrote, so you can react to it. In DPO no new text is ever written, so there's nothing to measure mid-run except the log-ratios on the fixed dataset. The leash's length is chosen once, by β.)*

Connects to: `../lecture-3-policy-gradients/02-wiring-rlhf-for-language-models.md` (the KL penalty); later `../lecture-9-over-optimization/` and `../lecture-10-regularization/`.

---

## The whole section in four lines

1. **DPO only sees the margin** between chosen and rejected log-ratios, not their absolute levels.
2. **Likelihood displacement:** the model can widen the margin by pushing both down (rejected faster). The chosen answer gets less likely and the probability flows to responses outside the data, possibly off-distribution. Example: chosen −50 → −53, rejected −48 → −60, loss 0.693 → 0.341.
3. **Fixes:** add an SFT term on chosen, or use Cal-DPO / AlphaPO. Watch `rewards/chosen` in your logs.
4. **β is a static KL:** it sets one fixed target distance from π_ref for the whole run. Online RL instead applies a per-batch KL penalty that can even be adjusted during training.
