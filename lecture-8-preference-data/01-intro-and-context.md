# Intro & Context

Notes on *Preference Data: The Most Opaque Part of Post-Training | Post-Training Course, Lecture 8* (Nathan Lambert), 00:00–07:34. Based on `source/lecture-8-preference-data-slides.md` and `source/lecture-8-preference-data-transcript.md`. The first file of the lecture. See `00-map.md` for where preference data enters the pipeline, an example datapoint, and the history timeline.

---

## Where this fits

Lectures 3–5 were about **optimizing** a reward: policy gradients, PPO/GRPO, and reasoning models trained on **verifiable** rewards. Lecture 8 steps back to ask **where the reward comes from when nothing can be verified.** The answer is preference data: people (or models) comparing two answers and picking the better one.

This lecture covers two chapters of the RLHF book:
- **Chapter 10:** a history and philosophy of what a "preference" even is (`02`).
- **Chapter 11:** preference data in practice: interfaces, formats, contracts (`03`), and open problems (`04`).

Lambert's framing: this material "at one point was very central to AI safety", and the problems are "still kind of unsolvable and interesting", but "almost nobody is talking about [them] today." He wants to point people down "rabbit holes."

The one idea to hold onto:

> **Preference data is how you get a training signal when there's no right answer. It's collected *before* RL, gets compressed into a single number by a reward model (or used directly by DPO), and that compression is imperfect in ways that come from deep, old problems about what "preference" means.**

---

## Slide: "How do we hillclimb on capabilities we don't really know how to score?"

A section-break slide with the core question of RLHF. "Hill climbing" means making a number go up step by step. In RLVR (lecture 5), the number is "fraction of math answers right." For "write a nicer poem," there's no such number. Preference data is "the entry point" that creates one. It "led to things like DPO" (`../lecture-6-dpo/`) and "is still used heavily today for reward modeling."

Lambert dates this question to "about 2018, which led to the modern RLHF literature."

---

## Slide: "Is it easier to write a good demonstration or decide between a good and a bad example?"

The intuition behind every preference pipeline. Lambert's example: if he asks you to *write* a great poem about a train struggling through a storm, that's hard. If he hands you a great poem and a bad one and asks which is better, that's easy.

| | Demonstration (SFT data) | Comparison (preference data) |
|---|---|---|
| What the person does | Writes the ideal answer | Picks the better of two |
| Skill needed | Must be able to **generate** quality | Only needs to **recognize** quality |
| Speed | Slow (minutes per answer) | Fast (seconds, sometimes) |
| Ceiling | Can't exceed the writer | Can point past the labeler's own ability |

*(My note: a film critic can't direct a film, but can reliably tell you which of two films is better. Preference data uses humans as critics, not directors.)*

Pipelines are designed "to leverage that intuition in a productive way and kind of manage the trade-offs that come with this kind of imprecise tool."

---

## Slide: "Which poem is better? Which model?"

Two poems for the same prompt, *"Write me a short poem about an optimistic goldfish,"* from Claude 3.7 Sonnet and GPT-4o (February 2025). Shown as **A** and **B**:

- **A:** tidy rhyming couplets: "A little goldfish, bright and bold, / Swam in circles, dreams untold..." ending "For in her heart, the ocean grew."
- **B:** looser, with a joke about goldfish memory: "Three seconds of memory, they say, / Yet each circuit feels brand new and gold..." The rhyme scheme wobbles.

Lambert admits he no longer knows which model wrote which ("if you are a model whisperer... I would love to know"). The point: **"it's a subtle question of which poem is better. And if you poll a bunch of people, you will get a distribution of answers."**

*(My addition, a small numeric picture.)* Say you show this pair to 10 labelers and 6 pick A. What do you train on?
- If each pair is labeled once, you might get "A wins" or "B wins" depending on who happened to label it. That's **noise**.
- The "truth" is closer to "A wins 60% of the time." That's exactly the kind of number the Bradley–Terry model tries to capture (`02`): a 60% win rate corresponds to a small reward gap between A and B.
- If the 4 people who picked B were all poets, maybe that's **signal**, not noise. That question returns in `04`.

---

## Slide: "When preference replaces correctness"

SFT and RLVR both had a notion of **correct** (the human-written answer, the passing test). "Preferences are what we reach for when there is **no right answer**": which poem, which explanation is clearer, which tone is kinder.

The slide's italic line: preferences "came to be the core of RLHF, as the alignment community looked to optimize for complicated objectives like 'human values.'"

Lambert stresses one thing he says he "should repeat in this lecture": **preference tuning, "whether DPO or full RLHF, has been very effective at improving capabilities."** This "subtle notion of preference can still make the models clearly smarter on performance benchmarks." So it's not just about style or politeness.

Connects to: `../lecture-5-reasoning-models/01-introduction-and-core-concepts.md` (the RLHF vs. RLVR slide). Same algorithms, different reward source. This lecture is about the RLHF column's reward source.

---

## Slide: "This lecture"

The plan:
1. A short **history** of preferences, and why it's an **imperfect** formulation (chapter 10) → `02`
2. **Preference data**: the trade-offs in practice (chapter 11) → `03`
3. **Open questions** (some shared with synthetic data, `../lecture-7-synthetic-data/`) → `04`

Lambert calls it "a less post-training-heavy discussion," and says RLHF remains "a great place for academics to work," which is "deeply part of why I think having an academic textbook on RLHF is still very valid."

---

## Slide: "The paper behind the first half of this lecture"

*The History and Risks of Reinforcement Learning and Human Feedback* (Lambert and two colleagues, 2023). He wrote it in summer 2023, while moving from Hugging Face to the Allen Institute (Ai2). He thinks it's "underrated."

| Claim | What it means |
|---|---|
| **History:** RLHF is where philosophy, economics, optimal control and RL meet | Each field brought its own idea of what a "preference" is |
| **Risk:** RLHF treats **costs, rewards and preferences** as interchangeable | They're different kinds of things (`02`, "Costs ≠ rewards ≠ preferences") |
| **Idea:** we inherit RL's optimizers and setup **without its convergence guarantees** | RL theory promises things only for a fixed, well-defined reward. A learned reward model isn't one |

Lambert's summary: it's about "the type of mental shortcomings that can come about when you're building on tools that are derived from these complex fields without... understanding what the problem formulation actually means."

---

## Slide: "Aside: The 'objective mismatch' lens"

This comes from Lambert's PhD work on **model-based RL**.

**In model-based RL** *(my note: background for readers new to it)*: instead of learning purely by trial and error, the agent first learns a **dynamics model** f, a predictor of "if I'm in state s and do action a, what state comes next?" Then it plans or trains a policy using that model as a simulator. The natural move is to make f as accurate as possible.

Lambert's thesis: **"the best policy with a dynamics model is not necessarily the most accurate dynamics model."** The model is trained for prediction accuracy, but you care about control performance. Those are different targets.

**The same mismatch in RLHF, "one level up":**

```
what we train the reward model for:    classify preference pairs correctly (accuracy on held-out pairs)
what we actually care about:           the policy trained against it is good
```

"The best reward model for making downstream performance good is not necessarily the one that is the most accurate on your training data." Lambert was thinking about this while building **RewardBench** (a benchmark that scores reward models by pair-classification accuracy): "I've always known that the RewardBench-style reward models are very much proxy."

The slide's punchline: **"Reward-model accuracy (RewardBench-style) is a proxy for a proxy!"**

```
human intent ──► preference labels ──► reward model ──► RewardBench accuracy
(what we want)   (proxy #1: noisy,      (proxy #2:       (a measure of how well
                  biased, `04`)          fitted to #1)    proxy #2 fits proxy #1)
```

*(My addition, a toy example.)* Reward model X gets 75% accuracy on held-out pairs, but has learned "longer is better" as a shortcut. Reward model Y gets 72%, but doesn't care about length. Run PPO against each. The policy trained on X learns to ramble (lecture 3's reward hacking) and ends up worse, even though X "scored higher." That's objective mismatch.

The plot on the slide is titled "alignment ceiling" (the image isn't in the sources). "Every proxy is imperfect, and that's kind of a story that'll come up again and again in the next few lectures" (`../lecture-9-over-optimization/`, `../lecture-10-regularization/`).

Connects to: `../lecture-3-policy-gradients/02-wiring-rlhf-for-language-models.md` (the reference model and KL leash exist *because* the reward model is a hackable proxy).

---

## When does this happen?

Nothing in this file runs inside the training loop. Everything here is about the **stage before RL**:

```
SFT model ──► collect preference pairs ──► train reward model ──► [freeze] ──► RL loop uses it at step 3
                    ▲                            ▲
           "easier to judge              objective mismatch lives here:
            than generate"               trained for pair accuracy, used for policy quality
```

---

## The whole section in four lines

1. RLHF's core question: **how do you hill-climb on things you can't score?** Answer: compare two outputs instead of writing the ideal one, because **judging is easier than generating**.
2. Preferences are a **distribution**, not a fact. Different labelers disagree (the goldfish poems), and that's either noise or signal.
3. Preference tuning (DPO or full RLHF) **does improve capabilities**, not just style.
4. **Objective mismatch:** a reward model is trained for pair accuracy, but used for policy quality. RewardBench accuracy is a proxy for a proxy.
