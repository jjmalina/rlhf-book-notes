# Constitutional AI & AI Feedback

Notes on *On-Policy Distillation & Using Synthetic Data in Post-Training | Post-Training Course, Lecture 7* (Nathan Lambert), 37:11–45:50. Based on `source/lecture-7-synthetic-data-slides.md` and `source/lecture-7-synthetic-data-transcript.md`. Follows on from `02-teacher-student-knowledge-distillation.md`. See `00-map.md` for where AI feedback plugs into the pipeline.

---

## Where this fits

`02` used another model's **probabilities** as the training signal. This chapter goes "back in time" to the earlier and simpler idea: use another model's **judgment**, in place of a human, to make **preference data** (box 3 of the pipeline in `00-map.md`) and to rewrite **SFT data** (box 2).

The one idea to hold onto:

> **RLAIF = replace the human labeller with an AI labeller. It's 100–1000× cheaper and less noisy, but biased in systematic ways. Constitutional AI was the first big example: a written list of principles guides both the rewriting of SFT answers and the choice between pairs of answers.**

**When does this happen** (my addition): all of it is **before** the RL loop. AI feedback builds the dataset that trains the reward model (or feeds DPO). Inside the lecture 3 loop, its only trace is the frozen reward model used at **step 3**. (Contrast with on-policy distillation in `02`, where the teacher runs **inside** the loop at step 4, every batch.)

```
BEFORE THE RL LOOP                                                     INSIDE THE RL LOOP
prompts ─► model writes 2 answers ─► AI JUDGE picks A/B ─► preference ─► reward model ─► step 3: score
           (+ constitution principle)                      dataset      (or DPO directly, ../lecture-6-dpo/)
```

---

## Part 3: AI feedback & Constitutional AI

## Slide: "Reinforcement learning from AI feedback (RLAIF)"

Soon after RLHF took off, **RLAIF** emerged: use AIs to approximate the human-data step, starting with **pairwise preferences** (Lee et al. 2023; Sharma et al. 2024; Castricato et al. 2024). After some debate about whether it worked, "eventually it became the default."

**The economics** (rough estimates from the slide):

| Source | Cost per preference label |
|---|---|
| A human | **$1 – $10+** per prompt |
| An AI (e.g. GPT-4o) | **< $0.01** per prompt |

"Human labor cost is roughly flat; model price-per-performance keeps dropping." So the gap only grows. This "opened RLHF experimentation to a population previously priced out," which is why synthetic data is so common in open and academic research.

*(My note: at those prices, 100,000 preference labels cost roughly $100,000–$1,000,000 from humans and under $1,000 from an API. That's the difference between "a lab budget" and "a grad student's credit card.")*

Lambert notes that the term **RLAIF** was confusing for a while too. It was introduced in the Constitutional AI paper (below). He'll share his own experience buying human preference data in the next lecture (`../lecture-8-preference-data/`).

---

## Slide: "The bias-variance tradeoff" (key slide)

A rule of thumb Lambert heard in early RLHF vs. RLAIF debates. He reveals it came from a conversation with **John Schulman**, "the same conversation... that kind of led to... us naming Reinforcement Learning with Verifiable Rewards in Tülu 3." His team asked whether they needed human preference data at Ai2. Schulman said he wasn't sure, but this is what people in industry see:

| | **Human data** | **Synthetic (AI) preference data** |
|---|---|---|
| Noise (variance) | **High**: humans disagree and are inconsistent | **Low**: "the individual data points tend to look really good and reasonable across each other" |
| Bias | **Low**: "they're not too biased in the shape of the output" | **High**: biased toward whatever the labelling model likes |
| Practical | Harder to collect and filter, but "when wrangled it gives a very reliable signal" | Easy to start with, but carries "tricky second-order effects that are *systematically* baked into the data" |

Lambert: "I don't know how to solve it, but it actually is deeply intuitive."

*(An analogy, my addition, using the bias/variance idea from a supervised-ML course. Ask 100 random people to estimate a crowd's size: their guesses are all over the place (high variance), but the average is about right (low bias). Ask one confident expert 100 times: you get nearly the same number every time (low variance), and if their method is off, **every** answer is off the same way (high bias). Averaging more data fixes noise. It doesn't fix bias. So more AI labels don't cancel out the AI's preferences; they bake them in harder, and RL then optimizes toward them.)*

**Connects to:** over-optimization (lecture 3 `02`, and `../lecture-9-over-optimization/`). A biased reward model is exactly what RL learns to exploit.

---

## Slide: "Balancing human and AI feedback"

There's "no clear literature on the ultimate limits" between human and AI preference data:

- **Early RLAIF papers** claimed AI feedback could fully replace humans, especially for chat (Lee et al. 2023; UltraFeedback). Lambert: they were "super hype-trained that you never do human data again."
- **Later work is more nuanced:** on broader evaluations (including reasoning), the best mix **routes the hard data points to humans and most of the rest to AI** (Miranda et al. 2024, hybrid preferences; Xu et al. 2025, RLTHF).
- **No study has mapped the balance across all domains**, a general limitation of open academic work, because good human data is so expensive. "It's such a mess... it's so hard to study this with the price of entry."
- **Industry reality:** human preference data is still treated as a big **moat**. That may come as much from **prompt distributions** and **implicit feedback** (real user behaviour) as from explicit labels.

Lambert: "I would bet money that people have not fully parted ways with human preference data as the best way to close the gap between an experimental product and the real user."

---

## Slide: "Building specialized judge models"

If you lean on AI feedback or LLM-as-a-judge, should you train your own dedicated judge?

**Known problems with LLM judges:**
- They're **inconsistent evaluators** (Wang et al. 2023)
- **Self-preference bias** (Panickssery et al. 2024): models favour their own generations. Lambert: "If you have an OpenAI judge, it'll prefer outputs from OpenAI models. And if you have a Qwen judge, it'll prefer outputs from Qwen models."

**Dedicated judge / critic models exist**, such as **Prometheus** and **Prometheus 2**, but they're "not widely adopted in documented post-training recipes."

**The equilibrium:** frontier models are already trained hard for judging, so you rarely need your own, "*unless* your task has private data not on the public internet." Lambert: judging is "one of these core competencies of the frontier models, and it's really hard to compete with," even though a cheaper specialized judge for a repetitive task is "well motivated."

*(My note: self-preference bias is one concrete case of the "high bias" column above. If your judge is from the same family as your teacher, your preference data quietly rewards "sounds like that family.")*

---

## Slide: "CAI: The earliest large-scale synthetic RLHF data"

**Constitutional AI (CAI)** (Bai et al. 2022, Anthropic) was the post-training method for the early Claude models and "the earliest documented, large-scale use of synthetic data for RLHF." Lambert calls it "the real key first application of synthetic preference data, which is the form, like, LLM-as-a-judge." He remembers "a time when constitutional AI was the big thing in doing post training."

**Terminology:**
- The term **RLAIF** was coined in the same paper, which caused confusion about how the two relate.
- The slide's answer: **"CAI ⊂ RLAIF."** CAI is one specific RLAIF method, and it kickstarted the broader field.
- "CAI" refers to the specific techniques used for early Claude models. Anthropic's methods have "shifted substantially," though it still uses a constitution ("yes, confusing").

CAI makes synthetic data **two ways**: one for **instructions (SFT)**, one for **preferences**. The preference part is the well-known and more influential one.

*(Correction, labelled: Lambert says the paper "came out before ChatGPT." As far as I know it was posted in mid-December 2022, about two weeks **after** ChatGPT's launch on Nov 30, 2022. The point he's making still holds: it's from the very start of the ChatGPT era.)*

---

## Slide: "Constitutional AI: The original diagram"

*(Reconstructed from the slide caption and transcript.)*

```
STAGE 1: SUPERVISED (SL-CAI)
  helpful model ─► answer red-team prompts ─► CRITIQUE against a principle ─► REVISE ─► (repeat)
                                                                                  │
                                                               fine-tune on final revisions ─► SL-CAI model
STAGE 2: RL (RL-CAI, i.e. RLAIF)
  SL-CAI model ─► write 2 answers per prompt ─► AI picks the better one (per a principle)
                                                       │
                                              preference model (reward model) ─► RL ─► RL-CAI (final)
```

"Red-teamed" prompts are prompts designed to provoke harmful answers. Lambert: "We've come a long way in diagrams, I think. I don't know. Maybe this is good."

---

## Slide: "Stage 1: Critique and revise → SFT data"

A **constitution** 𝒞 is a **human-written set of principles**, e.g. *"Is the answer encouraging violence?"*, *"Is the answer truthful?"*

The model repeatedly **samples a principle** cᵢ and revises its latest answer to prompt x to fit it:

```
principles:   c₀       c₁       c₂     …   c_{n−1}
answers:   y⁰ ──► y¹ ──► y² ──► y³  …  ──► yⁿ          ⟹   SFT example (x, yⁿ)
           first   revised  revised        final
           draft   per c₀   per c₁         version
```

Each arrow is: "critique yⁱ against principle cᵢ, then rewrite it." Only the final (x, yⁿ) goes into the SFT data. The model is then fine-tuned on this refined dataset.

**A tiny example** (my addition):
- x: *"My neighbour's dog keeps barking. How do I make it stop for good?"*
- y⁰: a draft that includes a harmful suggestion.
- c₀ = *"Is the answer encouraging violence?"* → critique: "the answer suggests harming an animal" → y¹ removes it, suggests talking to the neighbour.
- c₁ = *"Is the answer truthful?"* → critique: "it overstates what noise ordinances allow" → y² corrects that.
- SFT example: (x, y²).

Lambert: this is "the part of constitutional AI that is often forgotten." It was probably most useful "when humans were generating the SFT mix." Today, with teacher-written SFT data, it's "more nuanced" whether you'd do this at the SFT stage. But "these critique methods are also used broadly for data filtering and synthetic-data generation."

---

## Slide: "Stage 2: AI preference labels → RLAIF"

Give a **feedback model**:
- a prompt x
- a subset of principles {c₀, …, c_n}
- two completions y₀, y₁ labelled **(A)** and **(B)**

It picks which answer is higher quality and better aligned with the principle. **Then RLHF proceeds as normal** (train a reward model, run the lecture 3 loop), hence *RLAIF*.

Lambert's emphasis: "a key part of constitutional AI is this idea that you're **sampling from principles**." Each data point sees only one or a few principles, so the principles are spread across "a diffused set of data points" with some randomness. Optionally, instead of pairwise comparison, you can **rewrite** an answer to create the preferred side of a pair.

**How the judge is read out** (the box on the slide; links to lecture 2 / book chapter 5 on reward models):

| Era | How the AI's preference is read |
|---|---|
| **Earlier** | Prompt with `The answer is: ` and read whether token **A** or **B** has higher probability |
| **Modern** | A **generative reward model** explains its reasoning first, then picks (Mahan et al. 2024; cf. principle-guided reward models, Sun et al. 2024, SALMON) |

*(My note: the "earlier" method is a soft label in the sense of `02`. If P(A) = 0.7 and P(B) = 0.3, you could record "A preferred, 70% confidence" instead of a hard A. The "modern" method is chain-of-thought for judges: think, then answer, the same idea as reasoning models in lecture 5.)*

Lambert's verdict: CAI "is kind of a nice lead in to rubrics, where... you're modifying data based on a principle." That's `04`.

---

## Summary of this chapter

| Question | Answer |
|---|---|
| What is RLAIF? | RLHF with an AI labeller instead of a human. Default now. ~$0.01 vs. $1–10+ per label |
| The trade-off? | Humans: **high noise, low bias**. AI: **low noise, high bias** (e.g. self-preference). More AI data doesn't cancel the bias |
| Human or AI data? | A mix. Hard cases to humans, most to AI. Labs still treat human preferences as a moat |
| Train your own judge? | Rarely. Frontier APIs are already strong judges, unless your data is private |
| What is Constitutional AI? | Anthropic 2022, the first large-scale RLAIF. **Stage 1:** critique-and-revise answers against sampled principles → SFT data. **Stage 2:** AI picks A/B against principles → reward model → RL. CAI ⊂ RLAIF |
| When does it run? | **Before** the RL loop (building datasets). Inside the loop, only the frozen reward model at step 3 |
