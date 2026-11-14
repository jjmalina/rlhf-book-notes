# A Brief Overview of Preference Data

Notes on *Preference Data: The Most Opaque Part of Post-Training | Post-Training Course, Lecture 8* (Nathan Lambert), 20:17–31:11. Based on `source/lecture-8-preference-data-slides.md` and `source/lecture-8-preference-data-transcript.md`. Follows on from `02-a-short-history-of-preferences.md`. An example datapoint and the pipeline diagram are in `00-map.md`.

---

## Where this fits

`02` was the theory of why "preference as a number" is shaky. This file is RLHF book chapter 11: **how preference data is actually made**. Lambert calls it "a quick survey on preference data practices," mostly visual: interfaces, formats, and what buying human data is really like.

The one idea to hold onto:

> **There are many ways to ask for a preference (A/B, Likert scale, thumbs, pick-from-a-grid), but nearly everyone ends up with (prompt, chosen, rejected) pairs. Getting good pairs is expensive, slow, relationship-driven, and almost entirely undocumented in public.**

---

## When does this happen?

All of it happens **before** the RL loop (or before DPO), and much of it happens **over weeks**:

```
week 0        sign contract, write labeler instructions
week 1        small first batch: generate completions with your current model → labelers compare
              → you look at the data, find problems, revise instructions (much of it thrown out)
week 2–3      bigger batches; maybe swap in an improved model to generate the completions
...           ramp up to full volume
then          binarize → chosen/rejected pairs → train reward model (or run DPO)
then          RL loop: the reward model is frozen at step 3
```

(Reconstructed from the "Sourcing & contracts" slide caption and transcript.) Compare RLVR (lecture 5): there's no labeling phase. A verifier judges each completion **during** the loop, at step 3, fresh each batch.

---

## Slide: "Where we started: why collect preference data"

Back to `01`'s intuition: **it's far easier to judge than to generate.** "There are many cases where models can't at all generate a good answer, but can pick up on cues for which is better."

But: "collecting (and processing) any data like this well is the most **opaque** part of the modern post-training pipeline." (This is where the lecture's title comes from.)

Lambert: preference data "used to be more important when RLHF was the final step, but... is still definitely used to understand and improve the models today." Yet "there has been very, very little documentation on the whole process": collecting, documenting instructions, releasing data, and showing how it improves the model.

The best open example: **NVIDIA's HelpSteer datasets** (HelpSteer 2, HelpSteer 2-Preference, HelpSteer 3), released with their methodology, behind the open **Nemotron** models. "There should really have been many instances of this given how core the technology was... to only have one maybe... really showcases how parts of the literature can get left behind."

---

## Slides: "Example data collection interfaces" (five interfaces)

The interface images aren't in the sources; these are reconstructed from captions and transcript. The slide's lesson: **"Every interface shapes the preference it captures."**

| # | Interface | Who | What the labeler does | Signal it gives | Used for |
|---|---|---|---|---|---|
| 1 | **Research collection** | Anthropic, early Claude (Bai et al., 2022) | Read the whole conversation, compare two responses A/B on a **Likert scale** at the bottom, optionally add metadata | Pair + strength | Training (reward models) |
| 2 | **A/B test in production** | ChatGPT | "Which response do you prefer?" between two beta-model answers | Pair from real users | Usually evaluation, sometimes training |
| 3 | **Pairwise with ties** | Chatbot Arena (now Arena, once LMArena) | Pick A, B, **tie**, or "both bad" | Pair, or tie | Public evaluation / leaderboard |
| 4 | **Single bit** | Ai2 demos, most products | Thumbs up / down on one answer | One direction, no comparison | Evaluation, sometimes training |
| 5 | **Pick-from-many** | Image models (e.g. a grid of 4 images) | Click the one you like | "1 beats the other 3" | Training reward models for image generation |

What Lambert adds:
- **The ChatGPT A/B caption:** "The answers are very close: a reminder that preference data is noisy and hard to get exactly right."
- **On ChatGPT's user data:** he points to OpenAI's **sycophancy** blog post, which told us they had trained a reward model on this kind of user feedback, and "sycophancy emerged when that reward model got over-optimized in their training loop." *(My note: this is the April 2025 GPT-4o update that OpenAI rolled back. As far as I recall, the post-mortem names thumbs-up/down user feedback as the added reward signal, i.e. interface 4 rather than 2.)* This is the clearest public case of `04`'s biases getting amplified (`../lecture-9-over-optimization/`).
- **Thumbs:** "every model has this on it now."
- **Images:** "images are just so intuitive for human users to immediately know which one they like," and labs plug those clicks "immediately back into training."
- "Pretty much every permutation here has been tried."

---

## Slide: "Rankings vs. ratings"

"I've heard this so many times." The fundamental distinction:

| | **Rating** | **Ranking** |
|---|---|---|
| What | A score for **one completion in isolation** | A **relative comparison** of two or more |
| Example | "Paris is the capital of France, known for the Eiffel Tower." → **4 / 5** | A: "Paris." vs. B: "Paris, the capital since 987, home of the Eiffel Tower." → **B > A** |
| Typical role | **Metadata** | **What gets trained on** |

"In practice almost everyone trains on pairwise rankings, binarized to chosen/rejected for the Bradley–Terry loss... and keeps ratings on the side."

Two sources the slide cites:
- **Early Claude** used an **8-point** Likert ranking scale (Bai et al., 2022).
- **UltraFeedback** rates each completion separately, then **pairs a high-rated completion with a low-rated one**.

*(My addition, how UltraFeedback-style ratings become a pair.)* One prompt, four completions from different models, each rated 1–5 by a judge (UltraFeedback used GPT-4):

| Completion | Rating |
|---|---|
| c1 | 4.5 |
| c2 | 3.0 |
| c3 | 2.0 |
| c4 | 4.0 |

Take the best as **chosen** (c1, 4.5) and one of the others as **rejected** (e.g. a random lower one, c3, 2.0). Output: one row `(prompt, chosen=c1, rejected=c3)`, with 4.5 and 2.0 kept as metadata. The ratings decided *which* pair to make; the Bradley–Terry loss (`02`) only sees "c1 beats c3."

**Why not train on ratings directly?** *(My note.)* Absolute scores are inconsistent between labelers: one person's 4 is another's 3. Comparisons cancel out each person's personal offset, because both answers get judged by the same person at the same moment. That's the same reason Bradley–Terry only looks at the score *gap*.

---

## Slide: "What exactly is a 'Likert scale'?"

A **Likert scale** records a preference as an **ordered, graded** judgment: not just which answer wins, but **by how much**, on a symmetric scale, optionally with a neutral middle. Named after psychologist **Rensis Likert** (1932): the same "strongly agree... strongly disagree" survey tool, repurposed for pairwise preference.

**5-point with a tie** (LMArena-style): labelers can say "about the same."

| 1 | 2 | 3 | 4 | 5 |
|:---:|:---:|:---:|:---:|:---:|
| **A much better** | A better | *tie* | B better | **B much better** |

**8-point without a tie** (early Claude): forces a direction.

| 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **A>>>B** | | | A>B | B>A | | | **B>>>A** |

"How many points, and whether ties are allowed, are both design choices that change the data you collect."

*(My addition, binarizing Likert scores.)* With the 8-point scale, 1–4 means A wins and 5–8 means B wins:

| Likert score | chosen | rejected | Strength (margin) | What you might do |
|---|---|---|---|---|
| 1 | A | B | strong (3) | keep |
| 4 | A | B | weak (0) | keep, or drop as a near-tie |
| 5 | B | A | weak (0) | keep, or drop as a near-tie |
| 8 | B | A | strong (3) | keep |

On a 5-point scale with a tie, a "3" gives no pair at all and is usually dropped.

Lambert on how the strength got used: Likert scores were "used to kind of look at the data. At times, it was used for training, but it never really stuck." *(My note: one known use is Llama 2, which added a margin term to the Bradley–Terry loss so that "much better" pairs had to be separated by a bigger score gap.)* Still, "this is where a lot of the scores and data you'll see in the literature was downstream of."

---

## Slide: "Structured (synthetic) preference data"

In domains with structure, you can **build preference pairs automatically**, with no human labeler:

| Domain | How to make the pair | Who decides the label |
|---|---|---|
| **Math** | A correct solution ≻ an incorrect one | An answer checker (like a lecture 5 verifier) |
| **Instruction following (IFEval-style)** | Sample once **with** the constraint in the prompt and once **without**. Prefer the one that obeys it | A Python check of the constraint. "The label is free" |

The slide's example. Prompt: *"Describe the ocean. **Respond in all lowercase.**"*

| | Text | Check `text == text.lower()` |
|---|---|---|
| **Chosen** (sampled with the constraint) ✓ | "the ocean covers most of our planet, a restless sheet of salt water..." | passes |
| **Rejected** (sampled without the constraint) ✗ | "The ocean covers most of our planet. It is a restless sheet of salt water..." | fails |

"In these narrow domains, structured pairs can beat quality-judged preferences" (cited to Tülu 3, Lambert et al. 2024). It's "a simple form of synthetic preference data" (`../lecture-7-synthetic-data/`).

Lambert's caveat: "That has some gray area because sometimes the constraints are so hard that they make the quality of the actual content really contrived." But as a heuristic, building comparisons "based on a simple Python check is really useful."

**How this differs from RLVR (lecture 5):** both use a checker. In RLVR the checker runs **during** RL, at step 3, as the reward. Here it runs **before** training, to produce a fixed dataset of pairs for a reward model or DPO.

---

## Slide: "Beyond pairwise preferences"

"The pairwise comparison is a convention, not a law." Alternatives:

| Alternative | What the label is | Notes |
|---|---|---|
| **Directional / single-bit** | Thumbs up / down on one answer | Trained with **KTO** (Kahneman–Tversky Optimization, Ethayarajh et al. 2024). "I've heard [it] is actually pretty effective in industry" |
| **Token-level / fine-grained** | Mark specific tokens or spans as good or bad | Integrated into a reward model |
| **Natural-language feedback** | A written critique instead of a label | |

"Richer signal, harder collection (and not often used extensively in practice)." Lambert: these "have all faded away, kind of like process reward models faded away," because they're too hard to collect or don't show benefits at scale.

*(My note: KTO's name comes from Kahneman and Tversky's prospect theory, the same S-shaped value function shown in `02`. Its loss treats gains and losses asymmetrically.)*

---

## Slide: "Sourcing & contracts"

The figure (reconstructed from the caption; image not in sources): a **typical multi-batch human-data contract, about $500K**. Delivery starts small and ramps up "as goals and methodology narrow," with **"much of the first batches thrown out."** Larger contracts vary a lot.

Lambert on how it works: "at the beginning, you're getting a little bit of data, and then you slowly ramp up as your alignment with the data you are collecting improves." You give the vendor "a model that you're gonna get comparisons from," then try to improve that model "by week two or three," so the data keeps improving.

The slide's points:
- **Access is relationship-driven.** Vendors are supply-limited and favor big budgets and known brands.
- **"Millions get spent and partly wasted."** Few teams have the bandwidth to fully use human data.
- **Contracts often forbid releasing the data openly.** One reason preference data is so opaque.
- It's "a far more complex industry today with environments, etc."
- Even for simple preference data, **"you need a robust, existing post-training recipe to plug it into."**

Lambert's warning, from doing this once: "this comes on really fast... you need to have a really established organization of post-training to be able to effectively add in data. And a lot of times people buy data and nothing happens." People had "bad experiences of just like, oh, you try it, and it doesn't go well the first time." **"Do not go into the data-buying side for your training unless your internal expertise and methodology is very established."**

---

## Slide: "Guiding data collection: Labeler instructions"

After a contract is signed, buyer and vendor agree on **detailed instructions** for every task, "normally never seen outside the lab."

- One of the first **public** examples: OpenAI's full **InstructGPT** labeler instructions (2022). Pages of guidance on ranking **helpful, truthful, harmless**: "the spec that actually shapes the labels."
- They were later **deleted**. Lambert recovered them and mirrors them on rlhfbook.com. "So I have unwiped it from the Internet. And if OpenAI comes after me, that would be great."
- He's "not sure if it was for preferences or for completions."

*(My note: the instructions are the real definition of "better" in your dataset. Two labs with the same prompts and different instructions would train different reward models. `04` asks whether that spec ever shows up in the final model.)*

---

## Slide: "A dataset we bought: No Robots"

On Hugging Face's **H4** team, Lambert's group commissioned human data the way labs do:
- **No Robots**: 10K expert human-written **demonstrations** (SFT data, not preferences), matching the InstructGPT task distribution. Bought from a vendor and **released openly**, which is rare. On Hugging Face at `HuggingFaceH4/no_robots`.
- From the era of the **Zephyr** models and the **Open LLM Leaderboard**.

**Why they released SFT data and not preference data:** they tried to buy both. Preference data was "really hard for our small team" because **"on-policy data is very important. So you need to collect preference data from a model that's in your family."** SFT demonstrations "don't have this on-policy constraint."

*(My note on why on-policy matters, in timing terms.)* Preference pairs teach the reward model to tell apart *the kinds of answers it will see during RL*. During RL (lecture 3, step 2), the completions come from **your** policy. If the pairs were generated by some other model with a different style, the reward model learned to judge the wrong distribution, and at step 3 it's scoring text unlike anything it was trained on. A human demonstration doesn't have this problem: it's just a target to imitate.

Lambert: No Robots was some of "the first high-quality human completion data released after InstructGPT."

---

## Slide: "Preference evals are now a standalone business"

In 2026, **Arena** (formerly LMArena, formerly Chatbot Arena) reached a **~$100M annualized revenue run-rate within ~8 months** of launching its enterprise A/B-testing offering. Most of it is fees from model companies to test their models in front of real users.

Lambert: "people love to [knock] it... 'you don't know who the users are'... But it is still providing signal that is very important to the AI industry." The same pairwise-comparison tool that trains reward models is also, separately, a major way models are **evaluated** (`../lecture-12-evaluation/`).

---

## The whole section in five lines

1. **Judging is easier than generating**, but collecting judgments well is "the most opaque part" of post-training. NVIDIA's HelpSteer is nearly the only fully documented open example.
2. **Interfaces vary** (Likert A/B, production A/B, ties, thumbs, pick-from-grid), and each shapes the preference it captures. Nearly everything is **binarized to (prompt, chosen, rejected)**, with ratings and Likert strength kept as metadata.
3. **Structured pairs** (correct ≻ incorrect, constraint-followed ≻ not) give free labels and can beat human-judged pairs in narrow domains. They use a checker **before** training, where RLVR uses it **during** training.
4. **Buying human data** is slow, relationship-driven, ~$500K per contract, with early batches thrown out. It only pays off if you already have a strong post-training recipe. **Preference data should be on-policy** (from your own model family).
5. The same pairwise tool now powers **evaluation as a business** (Arena, ~$100M run-rate).
