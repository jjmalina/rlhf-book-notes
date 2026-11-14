# Lecture 5 Map

A one-page map for Lecture 5: **a timeline of the models**, **the typical reasoning-model pipeline**, and **where each recipe trick lands in the training loop**. It's the lecture 5 companion to `../lecture-3-policy-gradients/00-training-loop-timeline.md` and `../lecture-4-implementing-rl/00-code-map.md`.

Covers: `01`–`04` (the whole lecture).

---

## The typical 2025 reasoning-model pipeline

```
PRETRAINING (the cake) ─── includes MID-TRAINING: extra math/code near the end
     │                     → sets the ceiling for RL (MiMo, finding #15)
     ▼
COLD-START SFT (the icing) ─── a small set of good reasoning traces, to set the <think> format
     │                         (or big SFT on distilled traces: OpenThoughts 3)
     ▼
LARGE-SCALE RLVR (the cherry) ─── GRPO / PPO / CISPO-style loss (lectures 3–4)
     │                            + filtering, format/language/length rewards (03)
     │                            + async actor/learner infrastructure (lecture 4 04)
     ▼
DISTILLATION ─── 2025 (R1): SFT smaller models on the big model's traces
                 late 2025 → standard: MULTI-TEACHER ON-POLICY DISTILLATION
                 (specialist teachers give dense per-token signal to one student)
     │
     ▼
(at deployment) TEST-TIME SCALING ─── think longer; majority voting; selector models
```

"R1-Zero" skips the SFT stage: pure RL straight from the base model.

---

## Model timeline

| Date | Model | Lab | Why it matters | Notes |
|---|---|---|---|---|
| 2022–24 | STaR, TRICE, VinePPO | various | The idea before it worked: train on correct answers | `02` |
| late 2024 | **Tülu 3** | Ai2 | **Coined "RLVR"**: PPO on math correctness, general model | `02` |
| Sep 2024 | OpenAI **o1** | OpenAI | Started the wave (train-time and test-time scaling plots) | `01` |
| **Jan 20, 2025** | **DeepSeek R1 / R1-Zero** | DeepSeek | **The catalyst.** Pure RL → chain-of-thought. SFT → RL → distill | `02` |
| Jan 20, 2025 | Kimi k1.5 | Moonshot | Same day. Curriculum, online filtering, progressive length. No open weights | `02` |
| Feb 18, 2025 | Open-Reasoner-Zero | community | **First fully open** data + code + weights. **PPO**, λ = 1, no KL | `02` |
| Apr 29, 2025 | Qwen 3 | Alibaba | New default base. Thinking toggle (a headache). On-policy distillation for small models | `02` |
| Apr 30, 2025 | MiMo | Xiaomi | Whole pipeline + intermediate checkpoints. Pretraining sets the RL ceiling | `02` |
| Jun 5, 2025 | OpenThoughts 3 | community | 1.2M open reasoning traces. **SFT alone gets strong reasoning** | `02` |
| Jun 16, 2025 | MiniMax-M1 | MiniMax | **CISPO** + **FP32 LM-head fix** (inference vs. trainer log-prob mismatch) | `02` |
| Jul 28, 2025 | GLM-4.5 | Z.ai | 355B/32B MoE. Early **agentic** focus. China's open-weight lead | `02` |
| Nov 20, 2025 | Olmo 3 Think | Ai2 | **Fully open lifecycle.** ~1-month, ~200-GPU open RL run | `02` |
| Dec 1, 2025 | DeepSeek V3.2 | DeepSeek | Reasoning merged into the main model and **into tool use** | `02` |
| Dec 15, 2025 | Nemotron 3 Nano | NVIDIA | 31.6B/3.2B hybrid Mamba-Transformer. SFT + RLVR + RLHF. Very open | `02` |
| Dec 16, 2025 | **MiMo-V2-Flash** | Xiaomi | **Multi-teacher on-policy distillation** becomes the standard | `02` |
| Jan–Apr 2026 | Kimi K2.5/K2.6, GLM-5/5.1, DeepSeek V4 Preview | Chinese labs | Long-horizon agents, multimodal, 1M context | `02` |
| Mar–Apr 2026 | Nemotron 3 Super, Arcee Trinity-Large-Thinking | American labs | Open, agentic, long-horizon tool use | `02` |

**The arc:** o1 → math/code reasoning (spring 2025) → agents and tool use (fall 2025) → long-horizon agent substrates (2026).

---

## Recipe tricks on the training loop

The step numbers match lecture 3's timeline and lecture 4's code map.

| When | Step | Trick (`03` #) | The lecture 3–4 concept it builds on |
|---|---|---|---|
| Before training | – | **#1 Offline difficulty filter** (keep 20–80% pass rate) | GRPO: all-same group → advantage 0 (L3 `05`) |
| Before training | – | #15 Mid-training sets the ceiling | The LeCun cake: RL needs a good base |
| Each batch | 1 sample | #2 Online filtering / curriculum | Same, tracking the moving 20–80% band |
| Each batch | 2 generate | #10 Progressive max length | Length bias (L4 `06`), inference cost |
| Each batch | 2 generate | #11 Inference bottleneck, #12 async | Actors/learners (L4 `04`) |
| Each batch | 3 score | #8 Format reward, #9 language reward, #10 length penalty | The reward function. Auxiliary rewards can be hacked |
| Each batch | 3 → 5 | #3 Zero-gradient filter + active sampling | Every batch fully useful, so a stable loss |
| Each batch | 4 ref log-probs | #4 KL: β = 0 in short RLVR, returning for long agentic runs | KL leash (L3 `02`) |
| Each batch | 5 advantages | #7 Normalize across the **batch**, not the group | Whitening (L4 `02`) |
| Each batch | 6 loss | #5 Asymmetric clipping (DAPO) | Clip range ε (L3 `04`) |
| Each batch | 6 loss | #6 **CISPO** | Cap vs. off switch (L3 `05`, L4 `03`) |
| Each batch | 6 loss | #7 Per-token loss aggregation (Dr. GRPO, DAPO) | Loss aggregation (L4 `06`) |
| Each batch | 6 (ratio) | FP32 LM head (MiniMax-M1, `02`) | ρ needs matching log-probs from the inference engine and the trainer |
| After training | – | #13 Test-time scaling (voting, selectors) | The pass@k vs. pass@1 gap (`01`) |

---

## Three answers to "which tokens deserve credit?"

This connects lecture 3's credit-assignment thread to what lecture 5 says became standard:

| Approach | Per-token signal from | Cost | Where |
|---|---|---|---|
| GRPO / RLOO (bandit-style) | none: one advantage copied to every token | cheap | L3 `03`, `05` |
| PPO + GAE | a learned **critic** | +1 model to train, fragile | L3 `04`. "Don't sleep on PPO" for agents (`02`) |
| **On-policy distillation** | a **teacher's** per-token probabilities | needs strong teachers | L5 `02` (Qwen 3, MiMo-V2-Flash); full treatment in `../lecture-7-synthetic-data/02-teacher-student-knowledge-distillation.md` |

---

## Glossary additions (new in lecture 5)

| Term | Meaning |
|---|---|
| **RLVR** | RL with verifiable rewards: a program checks the answer (math match, unit tests) instead of a learned reward model |
| **Reasoning model** | A model trained (mostly via RLVR) to write a long `<think>` chain before answering |
| **Inference-time / test-time scaling** | Better answers from spending more compute when answering (longer thinking, more samples) |
| **pass@1 / pass@k** | Chance one try is right / chance at least one of k tries is right (1 − chance all k are wrong) |
| **"Zero" (R1-Zero, ORZ)** | RL straight from a base model, with no SFT stage first |
| **Cold-start SFT** | A small SFT stage before RL that teaches the format/style of reasoning |
| **Mid-training** | Extra high-quality (math/code) data near the end of pretraining |
| **Distillation** | Training a student on a teacher's outputs (off-policy: SFT on teacher text, which minimizes **forward** KL. On-policy distillation minimizes **reverse** KL: lecture 7 `02`) |
| **On-policy distillation** | The student writes, the teacher scores every token with its probabilities, and the student matches them |
| **MOPD** | Multi-teacher on-policy distillation: specialist teachers per domain → one student |
| **MoE (total / active)** | Mixture-of-experts: many expert sub-networks, only a few used per token. "355B/32B" = total/active parameters |
| **Hybrid / toggleable reasoner** | One model with thinking on/off or adjustable effort |
| **Format / language-consistency reward** | Auxiliary rewards for extractable answers / no language switching mid-trace |
| **Progressive length extension** | Raise the max generation length over the course of training |
| **Difficulty filtering** | Keep prompts the model solves ~20–80% of the time (where GRPO has signal) |
| **Active sampling** | Drop zero-advantage groups and keep sampling until the batch is full |
| **LM head** | The final layer that turns hidden states into vocabulary logits. Run in FP32 to match inference and trainer log-probs |
| **Rubrics** | Graders that check a list of criteria: "softly verifiable" rewards between RLHF and RLVR |
