# Lecture 8 Map

A one-page map for Lecture 8: **where preference data enters the pipeline** (before any RL loop starts), **what one preference datapoint looks like**, **a timeline of the ideas behind "preferences"**, and the new vocabulary. It's the lecture 8 companion to `../lecture-3-policy-gradients/00-training-loop-timeline.md` and `../lecture-5-reasoning-models/00-map.md`.

Covers: `01`–`04` (the whole lecture).

This lecture has almost no math and no new algorithm. It's about the **data** that feeds the reward model (or DPO), where the idea of a "preference" came from, and why it's an imperfect formulation.

---

## Where preference data enters the pipeline

The key timing fact: **preference data is collected and used *before* the RL loop runs.** By the time lecture 3's step 1 executes, the preference data has already been turned into a frozen reward model (or has already been used up by DPO).

```
PRETRAINING
     │
     ▼
SFT (instruction tuning) ──► SFT model
     │
     ▼
┌─ PREFERENCE DATA COLLECTION (this lecture) ─────────────────────────────────────┐
│ a. pick prompts                                                                 │
│ b. generate 2+ completions per prompt, ideally from YOUR model family           │
│    ("on-policy": Lambert, `03`)                                                 │
│ c. humans (or an AI judge, lecture 7) compare them: Likert scale, A/B, thumbs   │
│ d. binarize → (prompt, chosen, rejected) rows, ratings kept as metadata         │
│ e. often in batches over weeks: refresh the model, collect again (`03`)         │
└──────────────────────────────────────┬──────────────────────────────────────────┘
                                       │ the preference dataset
                    ┌──────────────────┴───────────────────┐
                    ▼                                      ▼
   PATH A: train a REWARD MODEL                 PATH B: DPO directly on the pairs
   (Bradley-Terry loss, `02`)                   (no reward model, no RL loop)
   → reward model FROZEN                        see ../lecture-6-dpo/
                    │
                    ▼
   RL LOOP (PPO, or GRPO with a reward model; lectures 3–4)
     step 1 sample prompts
     step 2 generate
     step 3 score  ◄── the frozen reward model is the ONLY place the
     step 4 ref log-probs     preference data "shows up" inside the loop
     step 5 advantages
     step 6 loss
     step 7 update
```

So, in lecture 3's table of "the cast": the **reward model** row says "❌ Never changes (trained in an earlier stage)". **That earlier stage is this lecture's data plus a Bradley-Terry training run.** The human judgments don't get looked at again during RL. Only the reward model's compressed version of them does.

*(My note: some pipelines repeat the whole box: collect new pairs from the current policy, retrain the reward model, run RL again. Llama 2 did several such rounds. Each round is still "collect first, then optimize", so the timing picture is the same, just repeated.)*

---

## Preference data vs. lecture 5's verifiable rewards

| | **Preference data (RLHF / DPO)** | **Verifiable rewards (RLVR, lecture 5)** |
|---|---|---|
| Is there a right answer? | No: "which poem is better?" | Yes: `answer == 77`, tests pass |
| What you collect beforehand | Many (prompt, chosen, rejected) comparisons, from humans or AI judges | Only prompts plus their reference answers/tests |
| What scores completions at step 3 | A **learned reward model**, trained on the pairs | A **program** (verifier) |
| When the "judgment" is made | **Before** training, frozen into the reward model | **During** training, fresh every batch |
| Cost | Expensive, slow (contracts, weeks, `03`) | Cheap once the checker is written |
| Failure mode | Proxy of a proxy: biases (length, lists, sycophancy) get over-optimized (`04`, `../lecture-9-over-optimization/`) | Much harder to game, but only covers checkable domains |
| Middle ground | **Structured preference pairs**: use a verifier to *build* chosen/rejected (correct ≻ incorrect, `03`) | Rubrics ("softly verifiable") |

---

## What one preference datapoint looks like

*(My addition: an illustrative row in the format most open datasets use, built from the lecture's goldfish-poem and Paris examples. Field names vary by dataset.)*

```json
{
  "prompt":   [{"role": "user", "content": "What is the capital of France?"}],
  "chosen":   [{"role": "assistant", "content": "Paris -- the capital since 987, home of the Eiffel Tower."}],
  "rejected": [{"role": "assistant", "content": "Paris."}],

  "likert_score": 7,          // optional: on an 8-point scale, 1 = A>>>B ... 8 = B>>>A; 7 = "B clearly better"
  "margin": 2,                // optional: how strongly (derived from the Likert score)
  "chosen_rating": 4.5,       // optional: per-completion 1–5 ratings (UltraFeedback-style metadata)
  "rejected_rating": 3.0,
  "annotator_id": "a17",      // optional: who labeled it
  "source_model": "sft-v3"    // optional: which model wrote the completions (on-policy check)
}
```

What actually gets used:

| Field | Used by | How |
|---|---|---|
| `prompt`, `chosen`, `rejected` | **Everything** (reward model, DPO) | The core training signal: "chosen beats rejected for this prompt" |
| `likert_score` / `margin` | Sometimes | Filter out near-ties, or (my note) add a margin term to the reward-model loss, as Llama 2 did |
| `*_rating` | Usually **metadata only** | How UltraFeedback-style pairs are *picked* (high-rated vs. low-rated), and for analysis |
| annotator / source info | Analysis, filtering | Checking disagreement, bias, on-policy-ness |

Lambert: "In practice almost everyone trains on pairwise rankings, binarized to chosen/rejected... and keeps ratings on the side."

---

## Timeline: how "preferences" became a number to optimize

Two roots that merged into RLHF (`02`), then the data practice (`03`).

| Year | Who / what | Field | The idea | Notes |
|---|---|---|---|---|
| 1662 | **Port Royal Logic** (Arnauld & Nicole) | Philosophy | Decision quality = outcome weighted by its probability (proto expected value) | `02` |
| 1789 | **Bentham's hedonic (felicific) calculus** | Philosophy | All of life weighed on one common scale | `02`. Slide says "early 1800s" (see correction there) |
| 1931 | **Ramsey**, *Truth and Probability* | Philosophy / probability | Quantify preference *and* belief together | `02` |
| 1932 | **Likert** scale | Psychology | Ordered, graded survey answers | `03` |
| 1947 | **Von Neumann–Morgenstern** utility theorem | Economics / decision theory | If preferences obey 4 axioms → one utility function; rational = maximize expected utility | `02`. What RLHF leans on to justify a scalar reward |
| 1950 | **Arrow's impossibility theorem** | Social choice | No voting rule fairly aggregates individual preferences | `02` |
| 1952 | **Bradley–Terry** model | Statistics | Pairwise comparisons → scalar scores | `02`. The reward-model loss |
| 1957 | **Bellman**: MDPs, dynamic programming | Optimal control | Optimal solutions for well-defined systems | `02` |
| 1973 | **Sen**, *Behaviour, Choice and Values* | Economics | Choice ≠ preference | `02` |
| 1984 | **Hirschman**, *Against Parsimony* | Economics | Preferences over preferences | `02` |
| 1988 | **Sutton**: temporal-difference learning | RL | Credit assignment | `02` |
| 1992 | **Watkins**: Q-learning | RL | Learning values | `02` |
| 2000 | **Ng & Russell**: inverse RL | RL | Learn a reward from behavior; "oddly absent" from RLHF | `02` |
| 2009 | **Singh et al.**, "Where do rewards come from?" | RL | Rewards as evolved primary rewards; value functions as secondary ones | `02` |
| 2013–17 | **DQN**, **AlphaGo / AlphaZero** | Deep RL | "RL works" (on closed-form rewards) | `02` |
| ~2018 | Modern RLHF literature | ML | "Easier to judge than to generate" → pairwise preference data | `01` |
| 2022 | **InstructGPT** labeler instructions released; **Anthropic** HH interface (8-point Likert) | Industry | First public looks at how preference data is made | `03` |
| 2023 | **UltraFeedback**; **No Robots** (H4, bought and released); Lambert et al. *History and Risks of RLHF* | Open data / research | Open AI-rated pairs; rare open commissioned data; the paper behind `02` | `01`, `03` |
| 2024 | **KTO** (thumbs up/down); HelpSteer 2; social-choice-for-alignment position paper; Model Spec | Research / industry | Beyond pairs; open human prefs + method; spec → data | `03`, `04` |
| 2025 | HelpSteer 3; OpenAI sycophancy post-mortem | Industry | Over-optimized user-feedback reward → sycophancy | `03` |
| 2026 | **Arena** reaches ~$100M annualized revenue in ~8 months | Industry | Preference *evals* as a standalone business | `03` |

**The arc:** philosophers reduce wants to one number → VNM makes it rigorous → social choice and behavioral economics show where it breaks → control/RL builds optimizers for a single closed-form reward → RLHF glues the two together, treating "preference" as "reward" → in practice, preference data is opaque, expensive and biased, and nobody can audit whether the spec ends up in the model.

---

## How this connects to lectures 3–5

| Lecture 8 idea | Builds on | Where |
|---|---|---|
| Preference data → reward model → step 3 | The frozen reward model in "the cast" | `../lecture-3-policy-gradients/00-training-loop-timeline.md` |
| Reward model as an imperfect proxy (objective mismatch, `01`) | Reward hacking / over-optimization; why the KL leash exists | `../lecture-3-policy-gradients/02-wiring-rlhf-for-language-models.md` |
| `rewards = reward_model(...)` | Step 3 in code | `../lecture-4-implementing-rl/00-code-map.md` |
| Verbosity bias in the data (`04`) | Length bias in the loss | `../lecture-4-implementing-rl/06-loss-aggregation-strategies.md` |
| Preference ≠ correctness | RLHF vs. RLVR | `../lecture-5-reasoning-models/01-introduction-and-core-concepts.md` |
| Structured pairs (correct ≻ incorrect) | Verifiers from RLVR, reused to build preference labels | `../lecture-5-reasoning-models/01-introduction-and-core-concepts.md` |
| "Inverse RL came back in on-policy distillation" (`02`) | On-policy distillation | `../lecture-5-reasoning-models/02-the-2025-cambrian-explosion.md` |
| DPO on pairs (path B) | – | `../lecture-6-dpo/` |
| AI-judged / synthetic pairs, on-policy data | – | `../lecture-7-synthetic-data/` |
| Biases amplified by optimization | – | `../lecture-9-over-optimization/` |

---

## Glossary additions (new in lecture 8)

| Term | Meaning |
|---|---|
| **Preference data** | Judgments of which of two (or more) completions to the same prompt is better |
| **Chosen / rejected** | The winning / losing completion in a pair. Also written y_w / y_l (w = "winner", l = "loser") |
| **Binarize** | Turn a graded judgment (Likert score, ratings) into a single chosen/rejected pair |
| **Rating** | A score for one completion in isolation (e.g. 1–5). Usually metadata |
| **Ranking** | A relative comparison of two or more completions. What gets trained on |
| **Likert scale** | An ordered, graded scale for "which is better and by how much" (e.g. 5-point with a tie, 8-point without) |
| **Margin** | How strongly one completion was preferred (from the Likert score) |
| **Utility function** | A single number that represents how much someone wants an outcome |
| **VNM theorem** | If preferences are complete, transitive, continuous and independent, a utility function exists |
| **Transitive / intransitive** | A > B and B > C imply A > C / they don't (rock-paper-scissors) |
| **Bradley–Terry model** | P(chosen beats rejected) = sigmoid(r_chosen − r_rejected). The reward-model training objective |
| **Objective mismatch** | Training a component (dynamics model, reward model) for its own accuracy, when what you care about is the final policy |
| **Social choice theory** | The economics field studying how to aggregate many people's preferences |
| **Arrow's impossibility theorem** | No voting rule satisfies a few basic fairness criteria at once |
| **Inverse RL** | Learning a reward function from observed behavior |
| **Costs vs. rewards vs. preferences** | Control-theory quantities / RL's scalar signal / human, relational judgments. Treated as interchangeable in RLHF; they're not |
| **Structured (synthetic) preference data** | Pairs built automatically: a correct answer ≻ an incorrect one, or a constraint-following answer ≻ one that ignores it |
| **On-policy preference data** | Pairs whose completions came from your own model (family), not someone else's |
| **KTO** | Kahneman–Tversky Optimization: trains on single thumbs-up/down labels instead of pairs |
| **HelpSteer** | NVIDIA's open human-preference datasets, released with their collection method |
| **Labeler instructions** | The vendor-facing spec of how to judge (e.g. InstructGPT's helpful/truthful/harmless guide) |
| **Prefix / verbosity / formatting / sycophancy bias** | Labels driven by the opening, length, markdown, or agreeing with the user, rather than quality |
| **Model Spec** | A public document of a model's intended behavior (OpenAI, 2024) |
