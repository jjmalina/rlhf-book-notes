# Lecture 7 Map

A one-page map for Lecture 7: **where synthetic data plugs into post-training**, **where on-policy distillation sits in the lecture 3 training loop**, and the new vocabulary. It's the lecture 7 companion to `../lecture-3-policy-gradients/00-training-loop-timeline.md`, `../lecture-4-implementing-rl/00-code-map.md` and `../lecture-5-reasoning-models/00-map.md`.

Covers: `01`–`04` (the whole lecture).

| File | Chapter | Time |
|---|---|---|
| `01-the-emergence-of-synthetic-data.md` | What synthetic data is, where it's used, the two meanings of "distillation" | 0:00–10:50 |
| `02-teacher-student-knowledge-distillation.md` | **The technical core**: from Hinton's KD to SFT to on-policy distillation, MOPD and self-distillation | 10:50–37:11 |
| `03-constitutional-ai-and-ai-feedback.md` | RLAIF, judges, Constitutional AI | 37:11–45:50 |
| `04-rubrics-as-rewards-and-conclusions.md` | Rubrics as rewards, wrap-up | 45:50–49:40 |

---

## The one idea

> **Synthetic data = any training data that another model helped make. It now feeds every box of the pipeline. The most technical form is on-policy distillation: the student writes, a teacher scores every token, and that score is plugged into the RL loop as a per-token advantage.**

---

## Where synthetic data plugs into post-training

```
                         ┌───────────── a model can do EVERY one of these jobs ─────────────┐
                         │                                                                    │
1. PROMPTS        seed prompts ──► LLM writes more like them (Self-Instruct)            01
                         │
2. SFT            prompt ──► TEACHER writes completion ──► filter ──► SFT the student    01, 02
                  (= "colloquial distillation" = off-policy distillation = SEQ-KD)
                  + Constitutional AI stage 1: critique & revise the completion          03
                         │
3. PREFERENCES    two completions ──► AI JUDGE picks A or B ──► reward model / DPO       03
                  (RLAIF; UltraFeedback; Constitutional AI stage 2)
                         │
4. RL REWARDS     completion ──► verifier (RLVR)                                         (L5)
                             ──► rubric + LLM judge → score (non-verifiable domains)     04
                             ──► TEACHER per-token log-probs (ON-POLICY DISTILLATION)    02
                         │
5. FILTERING      LLM judges quality, drops bad / repetitive samples (avoids "collapse") 01
```

Humans still matter at the edges: seeding the frontier, hard preference cases, and grounding evaluations (`01`, `03`).

---

## Three ways to train, on lecture 3's loop

The step numbers follow lecture 3's timeline: **1** sample prompts, **2** generate, **3** score, **4** reference log-probs, **5** advantages, **6** loss, **7** update. The full version with explanations is in `02` ("When does this happen").

```
                     OFF-POLICY DISTILLATION     ON-POLICY DISTILLATION         RL (GRPO / RLVR)
                     (SFT on teacher text)       (student writes, teacher       (lecture 3–5)
                                                  scores every token)
──────────────────────────────────────────────────────────────────────────────────────────────────
before training      TEACHER writes all          –                              –
                     completions → dataset
                     (frozen, on disk)
1 sample prompts     read rows from dataset      sample prompts                 sample prompts
2 generate           – (already written)         STUDENT generates              POLICY generates
                                                 (save student log-probs)       (save log-probs)
3 score              –                           – (no reward model)            verifier / RM → R
4 frozen forward     –                           TEACHER forward pass on the    (optional) π_ref
  pass                                           student's tokens → log π_T     forward pass
5 advantage          1 for every token           A_t = log π_T − log π_θ        (R − mean)/std,
                                                 (one per TOKEN)                copied to all tokens
6 loss               −log π_θ(teacher token)     −A_t · log π_θ(own token)      −A · log π_θ(own token)
7 update             student changes             student changes                policy changes
next batch           next rows (same old text)   NEW student text               NEW policy text
──────────────────────────────────────────────────────────────────────────────────────────────────
whose text?          teacher's                   student's                      policy's
per-token signal?    the token itself (1/0)      yes, from the teacher          no (bandit-style)
```

*(My note: on-policy distillation is the lecture 3 loop with the teacher sitting in π_ref's seat at step 4, no reward model at step 3, and the per-token "KL penalty" from step 5 promoted to be the **whole** advantage.)*

---

## How this connects to lectures 3–5

| Lecture 7 idea | Builds on | Where |
|---|---|---|
| Off-policy distillation = SFT on teacher text = minimizing forward KL | SFT as "advantage = 1" in the RL loss | L3 `00`, L4 `00` ("Loss in one line") |
| Exposure bias (train on teacher prefixes, test on own prefixes) | On-policy vs. off-policy data | L4 `04` |
| OPD advantage `log π_T − log π_θ` per token | The per-token KL `log π_θ − log π_ref` and its one-sample estimator | L3 `02`, L4 `00` (`per_token_kl = new_logps − ref_logps`) |
| OPD gives **dense, per-token** credit | "Which tokens deserve credit?": GRPO (none), PPO (critic), **OPD (teacher)** | L3 `03`–`05`, L5 `00` and `02` |
| MOPD (multi-teacher OPD) | MiMo-V2-Flash, the late-2025 standard | L5 `02` |
| Off- then on-policy strong-to-weak distillation | Qwen 3 small models | L5 `02` |
| SEQ-KD on reasoning traces | R1's "distill into smaller models", OpenThoughts 3 | L5 `02` |
| Rubrics as rewards | "Softly verifiable" rewards between RLHF and RLVR | L5 `01` |
| Judges / AI preferences | Reward model at step 3 | L3 `00` |
| Teacher and student must share a tokenizer | Token log-probs must line up (cf. the FP32 LM-head mismatch) | L5 `02` |

---

## Glossary (new in lecture 7)

| Term | Meaning |
|---|---|
| **Synthetic data** | "Any training data used for an AI model that is generated in some part from another AI model" |
| **Teacher / student** | The model whose outputs or probabilities are copied / the model being trained |
| **Distillation (colloquial)** | Train a weaker model on a stronger model's outputs. What "Chinese labs distill from GPT/Claude" means |
| **Knowledge distillation, KD (technical)** | Hinton et al. 2015: train the student to match the teacher's **full probability distribution** ("soft labels") |
| **Soft labels vs. one-hot** | A whole probability spread over the vocabulary vs. "the right token is X, probability 1" |
| **Word-level KD** | Match the teacher's full next-token distribution at every position of some existing text |
| **Sequence-level KD (SEQ-KD)** | Kim & Rush 2016: the teacher writes one output (beam search); the student does plain SFT on it |
| **Offline / off-policy KD** | Any KD where the training text came from the teacher (or a corpus), not the student |
| **Cross-entropy H(q,p)** | The standard next-token loss. Equals teacher entropy + KL(q‖p) |
| **Forward KL, KL(π_T ‖ π_θ)** | Averaged over **teacher** samples. "Mass-covering": the student must put some probability wherever the teacher does |
| **Reverse KL, KL(π_θ ‖ π_T)** | Averaged over **student** samples. "Mode-seeking": the student is only penalized where *it* puts probability the teacher dislikes |
| **Exposure bias** | Train/test mismatch: trained on the teacher's prefixes, tested on its own. Errors compound |
| **DAgger** | Imitation-learning method (Ross et al. 2011): the learner acts, an oracle labels what it should have done. Error grows ~εL instead of ~εL² |
| **On-policy distillation (OPD)** | The student generates; the teacher gives per-token log-probs on the student's own tokens; minimize reverse KL |
| **OPD advantage** | A_t = log π_T(a_t) − log π_θ(a_t): positive where the teacher likes the student's token more than the student did |
| **MiniLLM / GKD** | The two 2023–24 papers that started on-policy KD for LLMs (Gu et al.; Agarwal et al.) |
| **MOPD** | Multi-teacher OPD: per-prompt mixture of specialist teachers |
| **OPSD** | On-policy **self**-distillation: the teacher is the same model with a privileged hint in its context |
| **Privileged information** | Context (a hint, the bug, the answer) the teacher sees but the student won't at test time |
| **RLAIF** | RL from AI feedback: an AI, not a human, labels the preferences |
| **Constitutional AI (CAI)** | Anthropic 2022: critique-and-revise SFT data + AI preference labels, both guided by a written list of principles. CAI ⊂ RLAIF |
| **Constitution** | A human-written list of principles, sampled one (or a few) at a time |
| **LLM-as-a-judge** | Using an LLM to score or compare outputs |
| **Generative reward model** | A judge that writes out its reasoning before picking A or B |
| **Self-preference bias** | Judges favour outputs from their own model family |
| **Model collapse** | The worry that training on your own outputs narrows the distribution over generations. Avoided with filtering, mixing and diverse teachers |
| **Rubric** | A per-prompt checklist of weighted criteria (Hard Rule / Principle; Essential / Important / Optional / Pitfall) graded by an LLM judge into a reward |
| **Meta-prompt** | A prompt that tells an LLM how to *write the rubric* for each training prompt |
