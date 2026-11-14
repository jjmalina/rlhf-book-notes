# Rubrics as Rewards & Conclusions

Notes on *On-Policy Distillation & Using Synthetic Data in Post-Training | Post-Training Course, Lecture 7* (Nathan Lambert), 45:50–49:40. Based on `source/lecture-7-synthetic-data-slides.md` and `source/lecture-7-synthetic-data-transcript.md`. Follows on from `03-constitutional-ai-and-ai-feedback.md`. See `00-map.md` for the whole-lecture map.

---

## Where this fits

Constitutional AI (`03`) judged answers against **general** principles. Rubrics judge each answer against a checklist written **for that specific prompt**. They're how RL gets applied to domains with **no verifier**.

The one idea to hold onto:

> **A rubric is a per-prompt checklist, usually written by an LLM, that an LLM judge uses to score the policy's answer. That score is the reward at step 3 of the RL loop, so "non-verifiable" tasks can join the same scaled RL run as math and code.**

---

## Part 4: Rubrics, prompt-specific AI feedback

## Slide: "Why did rubrics become popular?"

Rubrics became a popular tool for **scaling RL on the long tail of domains**, and for domain-specific evaluation, "any other place domain expertise needs to be 'trained into' the models."

- **Extending RLVR** (lecture 5) to tasks *without* clearly verifiable answers. Lambert: they became popular after RLVR, when "people are trying to throw reinforcement learning at every problem, but there are a lot of problems that don't just have a verifiable answer."
- **The recipe:** write nearly-verifiable criteria for a prompt, generate multiple answers, and RL-update toward the best ones.
- Emerged **late 2024 → 2025** as LLM judges and synthetic-data practice matured. Also probably a function of RL becoming more broadly accessible to frontier post-training.
- Already delivering gains in **scientific reasoning** and **factuality** (Gunjal et al. 2025; Viswanathan et al. 2025, checklists; Rezaei et al. 2025, online rubrics; Liu et al. 2025, OpenRubrics).

Lambert: "At this point, it's like rubrics are used in every domain. If you have a specialized model or a specialized evaluation, the rubric is the way that you score this if you don't have a clear way to grade it."

**Connects to:** lecture 5 `01`, where rubrics were introduced as a **"softly verifiable"** middle ground between RLHF (a learned reward model) and RLVR (a program check).

| | RLHF | **Rubrics** | RLVR |
|---|---|---|---|
| Reward comes from | a reward model trained on preferences | an **LLM judge + a per-prompt checklist** | a program (answer match, unit tests) |
| Prompt-specific? | no, one model for everything | **yes** | yes (each problem has its answer) |
| Works without a right answer? | yes | **yes** | no |
| Hackable? | very (a learned proxy) | somewhat (the judge can be fooled; per-prompt rubrics help) | least |

---

## Slide: "A rubric example"

For a prompt with no single right answer, score against **tagged criteria** (OpenRubrics, Liu et al. 2025). Abbreviated:

```text
Prompt: As a museum curator, suggest five obscure artifacts for a
"Mysteries of the Ancient World" exhibit ...

Rubric:
1. Includes exactly five distinct artifacts.            [Hard Rule]
2. Each from a different culture and time period.       [Hard Rule]
3. Brief description of each artifact's significance.   [Hard Rule]
6. Communicates clearly and is well-organized.          [Principle]
8. Uses engaging language that stimulates curiosity.    [Principle]
```

- **[Hard Rule]** = atomic, must-pass checklist items (close to verifiable: you can count to five).
- **[Principle]** = softer quality criteria.
- The tags encode **priority** (numeric weights also work). "Subcomponents of a rubric contribute to the score."

**How it becomes a reward** (Lambert): "the student model generates a completion to the prompt, and a separate model will grade that [completion] based on the rubric. And then that grade is the reward signal in RL training."

**A worked scoring example** (my addition; the exact formula varies by paper). Say Hard Rules are pass/fail gates and each Principle is scored 0–1 by the judge:

| Answer | Rule 1 | Rule 2 | Rule 3 | Principle 6 | Principle 8 | Reward |
|---|---|---|---|---|---|---|
| A: five artifacts, all different cultures, described, clear, lively | pass | pass | pass | 0.9 | 0.8 | 0.85 (average of principles) |
| B: five artifacts, two from Egypt | pass | **fail** | pass | 0.9 | 0.9 | **0** (a gate failed) |
| C: six artifacts | **fail** | pass | pass | 0.8 | 0.7 | **0** |
| D: meets all rules, dry writing | pass | pass | pass | 0.8 | 0.3 | 0.55 |

With G = 4 answers to this prompt, GRPO (lecture 3 `05`) z-scores these rewards (0.85, 0, 0, 0.55) into advantages, so A and D get pushed up and B and C get pushed down. The rubric is just a way to produce R at **step 3**. Nothing else in the loop changes.

---

## Slide: "Per-prompt generation via a meta-prompt"

Rubrics are generated **per prompt**, "to ensure quality and robustness to over-optimization." For example, a **per-domain base rubric, refined per prompt by a supervising LM** (Gunjal et al. 2025). A truncated meta-prompt:

```text
You are an expert rubric writer for science questions ...
Choose 7-20 rubric items based on question complexity.
Each item: title (2-4 words), description (category prefix +
  what to look for), weight.
  - Essential : critical facts; omission invalidates the answer  (1-5)
  - Important : key reasoning / completeness                     (1-5)
  - Optional  : nice-to-have depth or style                      (1-5)
  - Pitfall   : common mistakes to penalize                    (-1,-2)
Output: a JSON array of {title, description, weight}.
```

"Real meta-prompts are long and tuned to the training setup." Note the **negative weights** for Pitfalls: a rubric can subtract points, not just add them.

**The layering** that makes this a synthetic-data story. Lambert: "you have a separate prompt to generate a prompt for a separate model, which judges a completion to your training prompt. I think I nailed it."

```
BEFORE THE RL LOOP (once per training prompt)
  meta-prompt + training prompt ──► RUBRIC-WRITER LLM ──► rubric (JSON), saved with the prompt

INSIDE THE RL LOOP (every batch)
  1 sample prompt (+ its rubric)
  2 policy generates G answers
  3 JUDGE LLM reads (prompt, rubric, answer) ──► score ──► R      ← the only new piece
  4–7 exactly as in lecture 3 (advantages, loss, update)
```

*(My note on "robustness to over-optimization": a single generic grader ("is this a good answer?") gives the policy one fixed target to learn to flatter, the lecture 3 `02` over-optimization problem. A rubric written for this prompt checks specific, content-level things ("exactly five," "different cultures"), which are much harder to satisfy by style alone.)*

**Lambert in practice:** for his new lab's domain-specific model, "one of the core things we know we're gonna do is we are going to create a rubric by talking to domain experts." It works "so long as you have a very high attention to detail" about what the synthetic outputs actually are. *(My note: so humans still matter here, as in `01`: experts seed the rubric criteria, and LLMs scale them to every prompt.)*

---

## Slide: "Where rubrics are going"

Rubric-based RL is a frontier of AI-feedback-driven training, expanding beyond its early uses:
- **Advanced instruction-following** (He et al. 2025, AdvancedIF)
- **Deep research** agents (Shao et al. 2025, DR Tulu, an Ai2 project)
- **Evaluating** research agents (Sharma et al. 2025, ResearchRubrics)
- **Long-form generation** with structured checklists (Ruan et al. 2025, ExpertLongBench)

"Very general tool!" The key property: "you can put this into the RL frameworks as a reward so that you could put these kind of non-verifiable domains in with your scaled RL run, which is running more agents or other tasks that have verifiers."

---

## Slides: Conclusions

### "Synthetic data is the single most common tool used by researchers (other than building infra) to make great models"

- The techniques will keep growing in complexity, "and it's super fun to see."
- "When I started writing this book, it was still a struggle to set up some synthetic data workflows!" There are still occasional cuts and slips, "but overall it's a well-known workflow now."
- There are many small open questions on how best to do it, often a domain-specific reflection of the technical problem at hand.

Lambert's verbal wrap-up sorts the lecture into two kinds of thing:

| Kind | Examples | Role |
|---|---|---|
| **Data as a core training optimization** | On-policy distillation (`02`) | "Outputs from the language model **are** the training signal" |
| **Ways of using LMs inside other training pipelines** | Constitutional AI (`03`), rubrics (`04`) | Making the preference data and rewards that RLHF, RLVR, DPO and reward models consume |

"This lecture really is like a fill-in-the-blanks of how else could you do this optimization, to go with all these previous discussions we've had on RL, DPO, reward models, and so on."

### "Course outline"

1. Introduction & training overview (Ch. 1–3)
2. IFT, reward models, rejection sampling (Ch. 4, 5, 9)
3. RL theory (Ch. 6, part 1)
4. RL implementation & practice (Ch. 6, part 2)
5. Reasoning (Ch. 7)
6. Direct alignment algorithms (Ch. 8)
7. **Synthetic data (Ch. 12)** ← this lecture
8. **Preferences & preference data (Ch. 10/11)**: next, "some philosophical backgrounds on preferences and preference data" (`../lecture-8-preference-data/`)

Questions go in the YouTube comments; Lambert will do round-up Q&A videos.

---

## The whole of lecture 7 in 8 lines

1. **Synthetic data is any training data another model helped make**, and it now feeds every stage: prompts, SFT completions, preferences, RL rewards and filtering. It won because it's 100–1000× cheaper and faster, and became possible once GPT-4-class models arrived. Humans still seed the frontier, ground evaluations, and supply the preference data labs treat as a moat.
2. **Model collapse is an outdated worry**: it only bites unfiltered single-model self-training. Filtering, mixing in real data and diverse teachers prevent it.
3. **"Distillation" has two meanings**: technical KD (match the teacher's full probability distribution) and colloquial (SFT on the teacher's text). Sequence-level KD shows the colloquial one is the technical one approximated, and cross-entropy = teacher entropy + KL shows it **minimizes forward KL** (mass-covering).
4. **Off-policy distillation suffers exposure bias**: trained on the teacher's prefixes, tested on its own, so errors compound (quadratically, by the DAgger analogy).
5. **On-policy distillation** lets the student write and the teacher score every token. That's **reverse KL** (mode-seeking), and in practice the per-token advantage **A_t = log π_T(a_t) − log π_θ(a_t)** drops straight into the lecture 3 RL loss. The teacher takes π_ref's seat at step 4, giving **dense credit** where GRPO gives one number per answer.
6. **Variants:** MOPD routes prompts to specialist teachers (MiMo-V2-Flash, DeepSeek-V4); OPSD uses the model itself plus a privileged hint as the teacher (Cursor Composer 2.5). All per-token KD needs a **shared tokenizer**, so it thrives inside one lab's model family.
7. **RLAIF** replaces human labellers with AI ones: humans are high-noise/low-bias, AI is low-noise/high-bias (e.g. self-preference). **Constitutional AI** (CAI ⊂ RLAIF) used sampled principles to rewrite SFT answers and to pick between pairs.
8. **Rubrics** are per-prompt checklists (often LLM-written via a meta-prompt) that an LLM judge turns into a reward at step 3, bringing non-verifiable domains into the same scaled RL runs as math and code.
