# Conclusions

Notes on *Implementing RL Algorithms for LLMs | Post-Training Course, Lecture 4* (Nathan Lambert), 52:08–53:36 (end). Based on `source/lecture-4-implementing-rl-slides.md` and `source/lecture-4-implementing-rl-transcript.md`.

---

## Slide: "Lecture summary"

From math to running code, the implementation details that matter:

| # | Topic | Where in these notes |
|---|---|---|
| 1 | **Core code patterns**: log-probs, masking, the REINFORCE/RLOO/PPO/GRPO losses | `01`, `02`, `03` |
| 2 | **Advantage estimation**: per-prompt baselines (RLOO, GRPO z-score) vs. per-token credit (GAE) | `01`, `02`, `03` |
| 3 | **Loss aggregation**: per-sequence vs. per-token vs. fixed-length changes training dynamics | `06` |
| 4 | **PPO's complexity budget**: value targets, critic clipping, double regularization | `02` |
| 5 | **Async training**: actor/learner splits, staleness budgets, distributed infrastructure | `04` |
| 6 | **Practical engineering**: batch size, monitoring, debugging, training recipes | `02`, `05`, `06` |

## Slide: "Resources"

| Book & course | Codebases | Key papers | Further reading |
|---|---|---|---|
| rlhfbook.com (the full book) | TRL (Hugging Face) | PPO (Schulman 2017) | Tülu 3 (a full open recipe) |
| rlhfbook.com/rl-cheatsheet | Open Instruct (Allen AI) | REINFORCE baselines (Ahmadian 2024) | DeepSeek R1 (reasoning RL) |
| github.com/natolambert/rlhf-book | veRL (ByteDance) | GRPO / DeepSeekMath (Shao 2024) | Unpacking DPO & PPO (Ivison 2024) |
| rlhfbook.com/code | OpenRLHF | GAE (Schulman 2015) | |

Lambert: "Please, please go look at the code. I'm begging people to help have some community action around this."

## Slide: "What's next"

- **Lecture 5: Reasoning** (book chapter 7), next, "after talking about RL so much"
- **Lecture 6: Direct alignment algorithms** (chapter 8), e.g. DPO, "which will be some more math and derivations"

His closing: "This one was a little bit harder... it's just kinda hard to look at code and explain it all clearly."

---

## The whole of lecture 4 in eight lines

1. **Every RL loss is `−(advantage × token log-probs)`, masked, reduced to one number, then `backward()`.** Algorithms differ in the advantage and the bounding.
2. **Log-probs:** forward pass → **shift by one** → `log_softmax` → `gather` the chosen token → **mask** to completion tokens. Always stay in log space.
3. **REINFORCE vs. RLOO is one line** (batch mean vs. leave-one-out per prompt). **GRPO is ~5 lines of group statistics**, with `unsqueeze(1)` broadcasting one advantage to every token.
4. **PPO caches** old log-probs, old values, ref log-probs and rewards **at rollout**, and **recomputes** new log-probs and values **every step**. GAE walks backward over tokens, and the critic regresses toward `advantages + old_values` (detached), with its own clipping.
5. **Clipping only activates when the model has drifted from π_old**, via minibatching or extra epochs. With K = 1 and no minibatches, it's a no-op.
6. **Real systems are slightly off-policy on purpose:** actors (vLLM) generate while learners train, with periodic weight syncs. The ratio and clipping absorb the staleness.
7. **The silent bugs:** a wrong mask (prompt or padding in the loss), reward on the wrong token, stale log-probs as current, NaN from zero-std groups or empty completions, and a missing `.detach()`.
8. **How you average per-token losses changes behavior**, especially answer length. Per-sequence (GRPO), per-token (DAPO) or fixed-length (Dr. GRPO). **Watch reward, KL, clip fraction, length and entropy, and read the outputs.** Big batches are the cheapest stabilizer.
