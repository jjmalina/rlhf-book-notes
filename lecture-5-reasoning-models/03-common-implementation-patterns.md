# Common Implementation Patterns

Notes on *The Rise of Reasoning Models | Post-Training Course Lecture 5* (Nathan Lambert), 34:54–42:42. Based on `source/lecture-5-reasoning-models-slides.md` and `source/lecture-5-reasoning-models-transcript.md`. Follows on from `02-the-2025-cambrian-explosion.md`. See `00-map.md` for the step-by-step map.

---

## Where this fits

This is the most practical part of the lecture: **15 recipe decisions that recur across 2025's reasoning-model papers.** Lambert's caveat up front:

> "There's a handful of models that do something, and that's a good sign that it's a pretty strong method. But **building language models is so path dependent** that you'll always have noise." Rarely does *every* model use a trick.

For each one, these notes give:
- **What** it is
- **When** it happens in the training loop. The step numbers follow `../lecture-3-policy-gradients/00-training-loop-timeline.md` and `../lecture-4-implementing-rl/00-code-map.md`: **1** sample prompts, **2** generate, **3** score, **4** reference log-probs, **5** advantages, **6** loss, **7** update
- **Why**, connected to lectures 3–4

The theme connecting most of them: **GRPO only learns from prompts where the G answers disagree** (lecture 3 `05`). Lots of these tricks are about not wasting compute on prompts that give zero signal.

---

## Recipe decisions (1/4): data and stability

### 1. Offline difficulty filtering ⭐ *(the most widely used trick)*

**What:** **before training starts**, take your candidate prompts, have the starting model answer each one **N times**, and **keep only prompts it gets right about 20–80% of the time.** Drop the ones that are too easy (~100%) or too hard (~0%).

**When:** before step 1, a one-time data-preparation pass.

**Why:** "where the gradient lives." From lecture 3 `05`: if all G answers to a prompt are right (or all wrong), every advantage is 0, so that prompt contributes **nothing** to the gradient but still costs generation compute. RL runs are expensive, so "you can remove prompts which are gonna be wasted compute before starting the run."

| Prompt pass rate | GRPO advantages | Worth training on? |
|---|---|---|
| 0% (always wrong) | all 0 | ❌ no signal, too hard for now |
| 20–80% | mixed + and − | ✅ the model sometimes succeeds, so there's something to reinforce |
| 100% (always right) | all 0 | ❌ nothing to learn |

Lambert: "probably one of the... most unanimous [tricks], with so many models implementing this." Used by Seed-Thinking 1.5, Open-Reasoner-Zero, Phi-4 Reasoning, INTELLECT-2, MiMo, Skywork OR-1 and Olmo 3.

*(Same pre-run stage: **decontamination**, which removes training prompts that overlap the eval benchmarks. See `../lecture-12-evaluation/04-can-you-trust-the-number-and-takeaways.md`.)*

### 2. Online filtering and curriculum

**What:** the same idea **during** training. As the model improves, prompts that were hard become easy. The data loader **skips prompts it now always solves**, and **saves harder problems for later**, when the model is ready.

**When:** step 1 (choosing which prompts go into each batch), continuously.

**Why:** the 20–80% band **moves as the model learns**. Offline filtering is a snapshot from the start; online filtering tracks it. Used by Kimi 1.5, Magistral, Llama-Nemotron, INTELLECT-2 and MiMo.

### 3. Zero-gradient filtering + active sampling (Olmo 3)

**What:** after scoring, **throw out any group whose G answers all got the same reward** (all advantages 0), and **keep generating until the batch is refilled** with groups that do carry signal.

**When:** between steps 3 and 5: score, check each group's spread, drop the dead groups, top up. In Olmo 3's async system (lecture 4 `04`), a prompt only enters the training batch if it has a non-zero GRPO gradient.

**Why:** otherwise a batch of 256 completions might effectively be 150, because the dead groups are dead weight in the average (lecture 4 `06`). With active sampling **every batch is fully useful**, which "makes loss nice and very stable." Lambert: the loss curve looks so smooth that "it almost makes it look like you need a learning rate schedule, sort of like pretraining."

**Filtering at three different times:**

```
before training:   offline difficulty filter (prompt pool → 20–80% pass-rate prompts)
each batch, step 1: online filter / curriculum (skip now-easy, defer still-hard)
each batch, 3→5:    zero-gradient filter + active sampling (drop all-same groups, refill)
```

### 4. KL penalty: dropped, now creeping back

**What:** single-turn math RLVR usually sets **β = 0** (no KL penalty, often no reference model at all). But **long-horizon, off-policy agentic training is bringing a KL/reference term back** for stability, e.g. **Cursor's Composer 2**.

**When:** step 4 (reference log-probs) and 5 or 6 (where the KL goes, lecture 4 `03`).

**Why:** lecture 3 `02`. KL exists to stop **over-optimization**: "the RL is really good at finding the easiest thing to move," which in RLHF meant stylistic tricks that turn into nonsense. Verifiable rewards are hard to hack, so early RLVR didn't need it. But **very long runs** can still drift, and "you want to make sure that when you do a really, really long RL run, that you have some grounding on what the model should stay near." Dropped by Magistral, Open-Reasoner-Zero and Skywork OR-1.

**Why β = 0 is survivable:** sampling its own outputs (on-policy RL) is already an implicit regularizer that keeps the model near where it started (RL's Razor). See `../lecture-10-regularization/03-why-rl-generalizes-better-than-sft.md`. What happens without enough grounding is shown in the GPT-4o sycophancy case, `../lecture-9-over-optimization/01-intro-and-why-we-care.md`. Why long agentic runs bring the reference back: `../lecture-11-tools-and-agents/04-training-for-tool-use-and-takeaways.md`. *(Lecture 10 also notes that some 2026 runs keep β = 0, which is in tension with the Composer 2 point.)*

### 5. Relaxed / asymmetric clipping (from DAPO)

**What:** use a **wider upper clip bound than lower** (e.g. ε_low = 0.2, ε_high = 0.28), so good tokens can rise further than bad tokens can fall.

**When:** step 6, in the clipped loss.

**Why:** **exploration.** With symmetric clipping, a rare but promising token (low probability) gets capped quickly, because ρ = 1.2 is a tiny absolute increase for something with probability 0.01. Widening the upper bound lets the model **lean into new, exploratory behaviors** faster. This is the asymmetric-ε preview from lecture 3 `04`. Used by Magistral and INTELLECT-2. "The list is actually longer than what I have on the screen."

---

## Recipe decisions (2/4): losses, normalization and rewards

### 6. CISPO

**What:** cap the importance-sampling weight (`clamp(ρ).detach()`) instead of switching tokens off when they move too much (lecture 3 `05`, lecture 4 `03`).

**When:** step 6.

**Why:** every token keeps getting gradient, which is more stable at scale. **ScaleRL** (a large study of RL scaling recipes) picked it as the best loss. Lambert: it "got picked up in all the scaling-for-RL papers as one of the most promising algorithms."

### 7. Normalization choices

How you average advantages and loss "quietly changes the objective." Two specific choices:

| Choice | What | Why | Who |
|---|---|---|---|
| **Normalize advantages across the whole batch**, not per prompt group | Divide by the batch's std rather than each group's std | Per-group std inflates groups with tiny spread, so a few very easy or very hard prompts (e.g. 7 of 8 right) can **dominate** the update | Magistral, MiMo |
| **Average loss over total tokens**, not per sequence | Per-token / fixed-length aggregation (lecture 4 `06`) | Removes the **length bias** of per-sequence averaging | Dr. GRPO (and DAPO) |

**When:** step 5 (advantage normalization) and step 6 (loss aggregation).

**Why:** this is lecture 4 `06` in practice. Lambert: these biases "change if the model is likely to be verbose or not, or to explore or not," and "a lot of papers kind of converged on doing this... per-token normalization."

### 8. Format rewards

**What:** an **extra reward** for producing a valid `<think>…</think>` block and an **extractable answer** (e.g. `\boxed{77}`).

**When:** step 3, added to the correctness reward.

**Why:** **usability, not correctness.** If the verifier or inference engine can't find the answer, the response is useless even if it's right. It teaches the model the output format the rest of the system expects. Used by DeepSeek R1, Open-Reasoner-Zero, Magistral and Skywork OR-1.

The evaluation side of the same problem (answer-extraction failures that make a model look worse than it is): `../lecture-12-evaluation/02-the-eras-of-post-training-evaluation.md`. Auxiliary rewards like #8–10 are the easiest to hack: "RL will always optimize the easiest reward" (`../lecture-9-over-optimization/01-intro-and-why-we-care.md`).

### 9. Language-consistency rewards

**What:** **penalize switching languages inside a reasoning trace.**

**When:** step 3.

**Why:** early reasoning models would start thinking in English, then **drift into Chinese or French** mid-trace. RL doesn't care which language gets the right answer, but people reading the trace do, and Lambert notes language switching is unstable. Used by DeepSeek R1 and Magistral.

**Format and language rewards together** show a general pattern: **the verifier checks correctness, and auxiliary rewards shape everything else you care about.** That's where reward hacking can creep back in (lecture 4 `06`'s forward pointer).

### 10. Length control

**What:** several ways to keep thinking **as long as needed, but no longer**:
- **Progressive length extension** (Kimi 1.5): start training with a **short maximum generation length** and raise it over time
- **A small length penalty** in the reward (INTELLECT-2)
- **Overlong filtering:** drop or handle completions that hit the max length, for throughput

**When:** step 2 (the max generation length) and step 3 (the length penalty).

**Why:** inference-time scaling means longer thinking scores higher, so RL tends to push length up. But you want "the model thinks for the shortest time possible to get the answer right," because long answers are slow and expensive to serve. Starting short makes the model "learn everything it can at the shorter lengths before letting it explore further." Lambert: "a ripe area for research that actually hasn't been documented as much."

---

## Recipe decisions (3/4): infrastructure and inference

### 11. Inference is the bottleneck

**What:** reasoning completions are **long and vary in length**. In Olmo 3, the **learner (training) GPUs sat idle ~75% of the time**, and runs used **5–14× more inference compute than training compute.**

**When:** step 2 (generation) dominates the wall-clock time.

**Why it matters:** this is lecture 4 `04`'s point at its most extreme. Generation is token-by-token, and thinking traces are thousands of tokens. One very long completion can hold up a whole batch.

### 12. Off-policy / async updates

**What:** actors generate continuously and the learner consumes. **Rollouts go slightly stale, but throughput jumps.** Used by Seed-Thinking 1.5 and INTELLECT-2. Olmo 3 has a deep dive (slides and recording linked in the source).

**When:** steps 1–3 run on actors while steps 5–7 run on the learner, overlapped (lecture 4 `04`).

**Why:** it's the direct fix for #11. The cost is slight off-policyness, handled by the ratio and clipping, plus care with the inference/trainer log-prob mismatch (MiniMax-M1's FP32 LM head, `02`).

### 13. Test-time scaling

**What:** spend more compute **when answering**, not training:
- **Majority voting:** sample many answers and return the most common one (DeepSeek R1, Phi-4)
- **A selector model:** a model that picks the best of several answers (DeepSeek-GRM)

**When:** at deployment/evaluation, after training. Not part of the training loop.

**Why:** **the pass@k vs. pass@1 gap shows the headroom** (`01`). If the model gets it right in 1 of 8 tries, a good way to choose among 8 tries captures some of that.

---

## Recipe decisions (4/4): findings

### 14. Text-only RL boosts multimodal

Text reasoning RL done **after** multimodal (vision) training **also improves vision tasks** (MiMo-VL, Magistral). Lambert: not too surprising, because post-training in narrow domains tends to generalize. For example, English-only safety RLHF carries over to other languages.

### 15. Mid-training sets the RL ceiling

**The math/code share in pretraining limits how far RL can go** (MiMo). It's the LeCun cake again: the cherry can only be as good as the cake under it. Lambert: "so far from surprising, but I would love to see more work documenting exactly how much of an impact that has."

---

## All 15 on the training loop

```
BEFORE TRAINING
   #1  offline difficulty filter (keep 20–80% pass-rate prompts)
   #15 (upstream) mid-training data sets the ceiling

EACH BATCH
 1. sample prompts ............. #2  online filtering / curriculum
 2. generate ................... #10 max length (progressive extension) · #11 the bottleneck · #12 async actors
 3. score ...................... #8  format reward · #9 language reward · #10 length penalty
    (drop dead groups, refill) . #3  zero-gradient filtering + active sampling
 4. reference log-probs ........ #4  KL: often skipped in RLVR, returning for long agentic runs
 5. advantages ................. #7  normalize across the whole batch
 6. loss ....................... #5  asymmetric clip · #6 CISPO · #7 per-token aggregation
 7. update ..................... #12 learner, with weights synced to actors

AFTER TRAINING
   #13 test-time scaling (majority voting, selector models)
   #14 finding: text RL also helps vision
```

---

## The whole section in five lines

1. **Most tricks are about feeding GRPO prompts with signal:** offline filtering (keep 20–80% pass rate), online curriculum (track the moving band), and zero-gradient filtering + active sampling (drop all-same groups and refill, which gives a beautifully stable loss).
2. **Algorithm tweaks:** KL dropped for single-turn RLVR (but returning for long agentic runs), asymmetric clipping for exploration (DAPO), CISPO (ScaleRL's pick), and batch-level advantage normalization + per-token loss to remove biases.
3. **Reward shaping beyond correctness:** format rewards (extractable answers), language-consistency rewards (no mid-trace language switching), and length control (progressive max length, small penalties).
4. **Infrastructure:** inference dominates (Olmo 3: learners idle ~75%, 5–14× more inference than training), so async actor/learner setups are standard. Test-time scaling (voting, selectors) uses the pass@k–pass@1 gap.
5. **Findings:** text RL helps multimodal, and pretraining/mid-training data sets the ceiling for RL. Every trick is path-dependent, so test it yourself.
