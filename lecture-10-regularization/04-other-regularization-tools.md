# Other Regularization Tools

Notes on *Regularization in RL, Why RL Generalizes, and Why SFT Forgets | Post-Training Course, Lecture 10* (Nathan Lambert), 25:15–31:33. Based on `source/lecture-10-regularization-slides.md` and `source/lecture-10-regularization-transcript.md`. Follows on from `03-why-rl-generalizes-better-than-sft.md`. Ends with the summary of the whole lecture.

---

## Where this fits

`01`–`03` were the KL story: the explicit penalty, and the implicit pull of on-policy sampling. This last part is a tour of **other things people add to control optimization**, and the 2026 shift in **what** regularization is protecting against:

| Era | What you keep the policy close to | Main tool |
|---|---|---|
| Early RLHF (2019–2023) | the **reference** model (start of the run) | KL penalty β·KL(π_θ ‖ π_ref) |
| Reasoning RLVR (2025) | (not much; verifiable rewards are hard to hack) | β = 0 + implicit on-policy regularization |
| Agentic RL (2026) | the **sampler**: the policy that generated the rollouts | **masking** tokens with large sampler-vs-trainer divergence |

---

## Slide: "Other regularization in the wild"

"Most of these are **scaffolding**: added to stabilize one setup, simplified away in the next model generation."

### 1. Pretraining next-token gradients (InstructGPT, "PPO-ptx")

Add to the RL objective:

```
+ γ · E_{x ~ pretraining data} [ log π_RL(x) ]
```

In words: alongside maximizing reward, also keep doing a bit of ordinary **next-token prediction on pretraining text**, weighted by γ. Purpose (InstructGPT's words): "to fix the performance regressions on public NLP datasets." Lambert: "they're trying to make sure the model they were training still had good next token prediction and didn't become dominated by the reward model." He hasn't "seen stuff like this recently."

*(My note: this is an SFT-style forward-KL term toward pretraining data, mixed into RL. It anchors to **text** rather than to a reference **model**.)*

### 2. NLL alongside DPO

```
L_DPO+NLL = L_DPO + α · L_NLL
```

DPO (lecture 6, `../lecture-6-dpo/`) only cares about the **gap** between the chosen and rejected answers. It's happy if both get less likely, as long as the rejected one drops faster. Adding NLL (the SFT loss on the **chosen** answer) "keeps the chosen text high-likelihood in absolute terms, not just relatively better" (Pang et al., Iterative Reasoning Preference Optimization).

*(Example, my addition: DPO alone could move log-probs from chosen −10 / rejected −12 to chosen −20 / rejected −25. The gap widened from 2 to 5, so DPO is pleased, but the chosen answer became far less likely. The NLL term penalizes that drop.)*

### 3. Margin loss for reward models (Llama 2)

```
L = −log σ( r_θ(y_chosen) − r_θ(y_rejected) − m(y_chosen, y_rejected) )
```

The normal reward-model loss (Bradley–Terry) asks: "is the chosen score above the rejected score?" The margin **m** raises the bar: the chosen score must win **by at least m**, and m is bigger when annotators said the two answers were **very** different. m comes from the annotators' rating deltas, "the Likert scales from Lecture 8" (`../lecture-8-preference-data/`). Lambert: it trains "a reward model that knew how far apart the two answers was."

### Why these come and go

Lambert's design philosophy here is worth keeping:

> "Adding complexity normally is a way to make the model behave just a bit better that you needed, but you want to revisit your recipe and make it simpler for the future model versions, because **the simpler ideas normally scale better**, even if that's just because you can scale your data better."

### When each one runs (my addition)

| Tool | Stage | Step |
|---|---|---|
| PPO-ptx (pretraining log-likelihood) | RL | step 6: an extra loss term on a separate minibatch of pretraining text, every training step |
| DPO + NLL | DPO (offline, no generation) | in the DPO loss, every step |
| Margin loss | **reward-model training**, before RL | the RM's loss. Affects RL only through the frozen RM in step 3 |

---

## Slide: "2026: The trust region / KL pen. is moving on"

"Tool use is changing what regularization has to do. In agentic recipes, the KL-to-reference penalty is disappearing."

- **GLM-5** removes it outright, "to accelerate RL improvement."
- **Kimi** doesn't use one either: "the K2 → K3 recipes ship with no KL penalty and **no reference policy at all**."
- **TMax** (Lambert's terminal-agent paper with colleagues, a fully open terminal-agent training data and models release): "a small KL **reduced the severity of collapse but lowered reward**, so the final recipe is **β = 0**."

What's actually breaking in agentic RL, per the TMax paper:
- **20+ turn trajectories** (very long rollouts with many tool calls)
- **async / partial rollouts** (lecture 4 `04`: the actor's weights lag the learner's)
- **train-vs-inference engine mismatch** (the same weights give slightly different probabilities in vLLM vs. the trainer, lecture 5's FP32 LM head issue)

Together these "make drift from the **sampler** the binding failure, not drift from init." In TMax, "instabilities increase past 10 assistant turns and were absent below 5."

Lambert: "in early RLHF, it was like training vs. reference, and now in RLVR for agents, it's like **sampler vs. trainer** is the thing that you want to be close."

*(My note: this squares with lecture 5's trick #4, which mentioned some long-horizon agentic runs bringing a KL/reference term **back** (Cursor's Composer 2). The field hasn't settled. Lambert's 2026 examples go the other way, toward β = 0 plus sampler-side controls.)*

---

## Slide: "A trust region on the sampling distribution" (DPPO)

**DPPO** (Divergence Proximal Policy Optimization, Qi et al., 2026) masks tokens by a **directly estimated divergence** between the rollout policy and the training policy (a "binary" total variation), instead of PPO's per-token ratio clip.

### The problem with ratios at low probability

PPO's ratio ρ = π_new(token) / π_old(token) (lecture 3 `04`). The slide: "The per-token ratio is a noisy one-sample estimate of that divergence." Lambert: "if these change a lot, there can be really big log ratios."

*(Figure reconstructed from the caption.)* For the same rollout tokens: per-token **ratios explode at low probabilities** (left panel), while the **TV divergence stays stable** (right panel).

**Numeric intuition (my addition).** "Binary" TV here roughly means: for the sampled token, how much did its probability change in absolute terms, |π_new − π_old|.

| Token | π_old | π_new | Ratio ρ | Binary TV ≈ \|change\| |
|---|---|---|---|---|
| rare token | 0.001 | 0.004 | **4.0** (way outside 0.8–1.2) | **0.003** (tiny) |
| common token | 0.50 | 0.70 | 1.4 | 0.20 |

The ratio says the rare token changed "4×!" and would get clipped. But in absolute terms almost nothing happened to the distribution. TV ranks them the right way round. This also links to lecture 5's DAPO clip-higher trick (asymmetric clipping), which existed because ratio clipping is too harsh on rare tokens.

### Masking, not clipping

Lambert corrected himself mid-sentence: "they're clipping individual tokens. And no. They're not clipping, but they're now **masking**."

| | **Clipping** (PPO) | **Masking** (DPPO, IcePop, Kimi) |
|---|---|---|
| What happens to a "weird" token | its gradient stops beyond the clip range (in the direction the advantage pushes) | the token is **dropped from the loss** entirely |
| Trigger | per-token ratio outside [1−ε, 1+ε] | estimated divergence (or log-ratio) outside a band |
| Protects against | moving too far from π_old in one batch | **sampler/trainer mismatch** and genuinely broken tokens |

Same family: **GLM's IcePop** and **Kimi's log-ratio interval** "do the same job: gradients masked, not clipped."

The slide's verdict: **"This is where the eye of regularization is today: algorithmic updates rather than KL penalties."** Lambert calls DPPO "somewhat popular, it hasn't really broken through," "still pretty new for me," and asks viewers who've used it to comment.

*(Correction: at ~29:00 the transcript has "we could go to lecture six... PPO chapter six." Lambert means **chapter 6 of the book** (the RL / PPO chapter, covered in lectures 3–4), not lecture 6, which was DPO.)*

### When does this happen? (timing)

```
2. generate with the SAMPLER (vLLM, maybe stale weights, async) → save log-probs  (= π_old / μ)
...
6. trainer recomputes log-probs with CURRENT weights             (= π_θ)
   per token: estimate divergence(sampler, trainer)
      inside band  → keep in loss
      outside band → MASK (contributes nothing to the gradient)
   loss → backward
7. update
```

No reference model is involved at all. Compare `01`: the old KL penalty compared the policy to π_ref (the **start of the run**). This compares it to π_old (the **sampler of this batch**). Lecture 3's two "old models" table (`../lecture-3-policy-gradients/00-training-loop-timeline.md`) is exactly this distinction: the 2026 regularization moved from the π_ref column to the π_old column.

---

## Slide: "Takeaways"

- The KL penalty is the explicit control: a **reverse KL**, estimated on the policy's own samples.
- The KL-regularized RL objective **is** a reverse-KL minimization, mode-seeking toward a reward-tilted reference. SFT is the forward direction, mass-covering toward the data.
- Even with no penalty, on-policy RL is implicitly regularized: **SFT memorizes, RL generalizes.**
- **"Verifiable rewards are much less prone to overoptimization than reward models!"** So regularization is changing and the KL penalty is playing a smaller role.

Lambert's summary: the KL penalty "came through RLHF to control the reward model over-optimization we talked about in the last lecture." SFT "has to kind of try to memorize things... and has some more numerical problems that lead to it forgetting a bit more."

---

## Slides: "The course so far" and "Thank you"

Lecture 10 = book chapter 15. Next (tentative): **Evaluation** (chapter 16), then hopefully **Basics of tool use** (chapter 13). Lambert: "I just have about three to four to five, depending on surprises, lectures left."

---

## The whole of lecture 10 in 8 lines

1. **Regularization answers lecture 9's over-optimization**: it controls how far one RL run moves the model.
2. **Explicit:** r = r_θ − β·KL(π_θ ‖ π_ref), a reverse KL computed from the policy's own samples. Reference log-probs are cached at rollout (step 4). PPO folds KL into rewards before advantages; GRPO recomputes it in the loss every step. β began as an adaptive controller chasing a target KL, then became a static 0.05, then often 0.
3. **KL direction = where the samples come from.** First slot = the sampler. Model in the first slot = reverse.
4. **RL with β > 0 is exactly reverse KL to the reward-tilted reference** π* ∝ π_ref·exp(r/β). **SFT is exactly forward KL to the data.**
5. **Forward = mass-covering** (the target picks where you're checked, so cover everything). **Reverse = mode-seeking** (you pick, so sit on a good peak).
6. **Backwards intuition, corrected:** SFT "covers" the **new data**, stealing mass from old modes, so it forgets. RL samples itself and can move one mode while leaving the rest. SFT memorizes (80.8% → 1.3% OOD), RL generalizes (80.8% → 91.8%).
7. **RL's Razor:** among equal solutions RL finds the lowest-KL one, forgetting tracks D(π_0 ‖ π), and on-policy data alone explains it. That's why **β = 0 works** in short RLVR (lecture 5).
8. **2026:** other add-ons (PPO-ptx, DPO+NLL, margin loss) were scaffolding. Agentic RL drops the reference entirely and instead **masks tokens** where the **sampler and trainer** disagree (DPPO, IcePop, Kimi). Regularization moved from "stay near the start" to "stay near the sampler."
