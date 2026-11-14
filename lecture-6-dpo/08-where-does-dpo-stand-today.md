# Where Does DPO Stand Today?

Notes on *Direct Preference Optimization (DPO) and Friends | Post-Training Course, Lecture 6* (Nathan Lambert), 40:33–42:44. Based on `source/lecture-6-dpo-slides.md` and `source/lecture-6-dpo-transcript.md`. Follows on from `07-dpo-implementation-details.md`. Ends with the whole lecture in eight lines.

---

## Slide: "Where does this leave us today?"

The puzzle: "If so many models have used DPO well and it's so simple, why does it seem like it comes up so infrequently — especially at the frontier?" Lambert: "a lot of models that I have worked on have used DPO. And there's literature showing that RL is better, but DPO is so simple."

---

## Slide: "Hypotheses for DPO's role today"

### Recent models that used DPO

| Model | When | How it used DPO |
|---|---|---|
| **Olmo 3** (Ai2) | Nov 2025 | DPO over **reasoning traces**, after SFT, "to boost performance in a simple pipeline" (with delta-learning pairs, `07`) |
| **SmolLM 3** (Hugging Face) | Jul 2025 | Same pattern: DPO over reasoning traces after SFT |
| **NVIDIA Nemotron 3** | Dec 2025 | *Mixed Preference Optimization*: DPO + Binary Classifier Optimization in one offline stage, on data scored by a **generative reward model** |
| **Liquid AI LFM2** | Nov 2025 | **Length-normalized** DPO (like SimPO's normalization, `06`) on **semi-online** data, **before a final RLVR pass** |

Lambert: "realistically we can classify these as non-frontier models." They use DPO "as a way to get solid performance."

### Where these sit in the pipeline *(my addition, from the table)*

```
pretrain → SFT → DPO (offline pairs; often over reasoning traces) → [RLVR, e.g. LFM2] → release
                  └─ the cheap, stable stage that gets a "good/solid" model
```

"Semi-online" is the middle ground from `07`: the pairs are regenerated from a recent model now and then, rather than every step (online RL) or never (pure offline).

### How Lambert sees it

- **DPO works well in "wonky, distillation-heavy recipes like Olmo"**: for example, a model with a "spikier distribution," trained on data from many teacher models. "Post training is not this super refined science yet. And DPO works really well when you're kind of in this early stage before your data pipelines get super, super refined."
- **It still works elsewhere**, but "most labs have the engineering resources to do other things with higher peak performance." As you "scale up the skill tree" (more human data, more refined data, more engineering), you "get more payoffs to doing this RL-style optimization and integrating your preference data in with other RL stages."
- **TL;DR: "DPO is a path to a good/solid model, but not to the best model.** And DPO may do less in *cleaner* training recipes." It's "often really good for getting it off the ground," and he'll "keep using it for experimenting with new types of models."

*(My note: that's the answer to `01`'s "DPO vs. RL" debate, "the answer is both." DPO for a fast, cheap, stable stage; online RL when you can afford it and need the last bit of performance. The 2025 reasoning recipes in `../lecture-5-reasoning-models/` put RLVR at the end for exactly this reason.)*

### Closing

"It is very important to understand DPO if you're going to understand post training." Next is **lecture 7**, where Lambert pulls the **synthetic data** chapter forward (`../lecture-7-synthetic-data/`). Questions go under the video. A Q&A video comes with this batch of lectures, which he's releasing in three or four batches over the summer.

---

## The whole of lecture 6 in 8 lines

1. **DPO (May 2023) made post-training accessible:** no reward model, no RL loop, one loss on preference pairs. Zephyr-Beta and Tülu 2 (fall 2023) proved it, with UltraFeedback's synthetic pairs as "the match."
2. **The derivation, in plain words:** the RLHF objective (reward − β·KL) has an exact answer, **π* = π_ref × e^(reward/β), renormalized by Z**. Turned around, **reward = β·log(π/π_ref) + β·log Z**: "your LM is secretly a reward model."
3. **Bradley-Terry only looks at reward gaps**, so the uncomputable Z cancels. The result: **loss = −log σ(β·Δlog-ratio(chosen) − β·Δlog-ratio(rejected))**, logistic regression on "which one wins."
4. **The gradient is weighted supervised learning:** +βw on the chosen response's log-prob and −βw on the rejected one's, where **w = "how wrong" the model currently is**. No reward model provides the weight. The model's own implicit reward gap does.
5. **Timing:** offline. Pairs are fixed before training, reference log-probs are cached once, and each step is two teacher-forced passes plus a backward. No generation, no scoring, no advantages, no π_old.
6. **Weaknesses:** it only sees the margin, so the chosen answer can get less likely (**likelihood displacement**), and β sets a **static** KL target for the whole run. Online RL can explore beyond the dataset and usually peaks a bit higher.
7. **A zoo of variants** (IPO, cDPO, ODPO; reference-free ORPO and SimPO) each change one line of the loss, but **the base model and the data matter far more than the algorithm**: on-policy completions and a clear chosen-vs-rejected gap (delta learning).
8. **Today:** DPO is "a path to a good/solid model, but not to the best model." It's used after SFT in open, distillation-heavy recipes (Olmo 3, SmolLM 3, Nemotron 3, LFM2), often before a final RLVR stage. Frontier labs mostly fold preferences into RL.
