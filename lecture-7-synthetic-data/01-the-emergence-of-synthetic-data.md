# The Emergence of Synthetic Data

Notes on *On-Policy Distillation & Using Synthetic Data in Post-Training | Post-Training Course, Lecture 7* (Nathan Lambert), 00:00–10:50. Based on `source/lecture-7-synthetic-data-slides.md` and `source/lecture-7-synthetic-data-transcript.md`. See `00-map.md` for where each kind of synthetic data plugs into the pipeline and the training loop.

---

## Where this fits

Lectures 3–5 were about the **algorithm** (policy gradients, GRPO, PPO) and what people built with it (reasoning models). Lecture 7 is about the **data** those algorithms eat, and the surprising fact that almost all of it is now written or judged by other models.

The lecture is "secretly a very dense lecture." It looks like a survey ("just train on the outputs of a strong model"), but the middle chapter (`02`) is a full derivation of **on-policy distillation**, which ties straight back into the lecture 3 loop.

The one idea for this chapter:

> **"Synthetic data is everything. Synthetic data is just an output from a model in any way."** It took over because it's cheap and fast, and it only became possible once models reached GPT-4 quality.

---

## Slide: "Why synthetic data took over"

When RLHF started (InstructGPT, early ChatGPT), human data was **the only** way to get good responses and reliable feedback. As models improved, that broke down.

| Reason | What the slide says | What Lambert adds |
|---|---|---|
| **Cheaper, faster iteration** | Lowers the price of a post-training experiment, so people priced out of human-data pipelines can join | "When so much data was bottlenecked on needing to get human signal, things just took a lot longer." Fast synthetic loops are what make today's **RSI** (recursive self-improvement) talk plausible |
| **A capability threshold** | Only worked once GPT-4-class models arrived. Llama 2 and GPT-3.5-Turbo couldn't reliably generate *or* supervise data. LLM-as-a-judge emerged in the GPT-3.5 → GPT-4 jump | Around 2023–24, open-weight GPT-4-level models "diffused" the capability, so now you can pick from many teachers |
| **The center of gravity** | Leading models *need* synthetic data to reach the frontier | Humans "provide signal at the absolute frontier and kind of can create a focal point." Everything else is synthetic |

He frames this personally: when he started writing the book there was "substantive debate" on whether synthetic data could do everything. "And now... synthetic data just does everything."

---

## Slide: "This lecture"

**Definition:** "Synthetic data is any training data used for an AI model that is generated in some part from another AI model. This is very broad!"

The plan:
1. The **roles** of synthetic data (this file)
2. **General distillation** with synthetic data (this file)
3. The path to **on-policy distillation**, the technical core (`02`)
4. **AI feedback** and **Constitutional AI** (`03`)
5. **Rubrics**, prompt-specific AI feedback (`04`)

Lambert on rubrics: they "had a flash in the pan in early 2026 as a crucial topic, and they've kind of quieted down. But everybody uses them now," for both training and evaluation. *(The slides date their rise to late 2024–2025.)*

---

## Slide: "Lecture 7: Where it sits"

This is **book chapter 12, "Synthetic data & CAI"**, in the "Data & preferences" part of rlhfbook.com, alongside chapters 10–11 on preferences (next lecture, `../lecture-8-preference-data/`). Lambert: "We're about halfway through the book."

---

## Slide: "Recall: Where synthetic data sits in a pipeline"

The post-training pipeline has many moving parts. Synthetic data from other LLMs now feeds **every box**:

| Box | How a model does it |
|---|---|
| 1. Collect / generate **prompts** | Give a model a small set of good prompts and ask for "similar things" |
| 2. Generate **completions** for SFT | A teacher writes the answers |
| 3. Collect **preferences** for RLHF | A model picks which of two answers is better (to train a reward model) |
| 4. Score **rewards** for RLVR | Verifiers, or "just use a judge model... as an actual reward for the environment" |
| 5. Filtering and cleaning | A model judges quality and drops bad samples |

The figure (reconstructed from the slide caption and transcript) is Lambert's usual multi-round post-training diagram: many rounds of feedback and optimization, with synthetic data now feeding most of the loop.

He admits on camera that he lacks a clean "what is synthetic data?" slide and will add one. His working definition: "Synthetic data is just an output from a model in any way."

*(My note: see the pipeline diagram in `00-map.md`. Every later topic in this lecture is one of these boxes: SFT completions and distillation (`02`), preferences via AI feedback (`03`), RL rewards via rubrics (`04`). On-policy distillation is special because it's a **reward** (box 4) built from a teacher's probabilities.)*

---

## Part 1: The roles of synthetic data

## Slide: "What synthetic data is used for"

"A single model is reused for many roles":

| Role | Seminal example | Lambert's comment |
|---|---|---|
| Generate new prompts from seed examples | **Self-Instruct** (Wang et al. 2022) | An LLM can generate similar high-quality prompts |
| Modify / expand existing prompts | – | – |
| Generate completions to prompts | NuminaMath (2024) | Early work (some from his Hugging Face days) doing synthetic completions from a GPT model, before reasoning models |
| Provide AI feedback → preference data | **UltraFeedback** (Cui et al. 2023) | "The first kind of AI preferences dataset to be really popular" |
| Filter completions for quality | Superfiltering (Li et al. 2024) | – |
| Verify answers as rewards for RL | (RLVR, lecture 5) | – |

**How reasoning models changed this:** before reasoning models, "you could just distill from the OpenAI API as a customer." Now you want the **reasoning traces**, and closed APIs hide them, so it "takes a bit more flexibility." (More on this two slides down.)

---

## Slide: "The balance between human and synthetic data"

Synthetic data didn't replace human data evenly:

| Stage | Who won | Lambert |
|---|---|---|
| **Instruction data (SFT)** | Synthetic, largely. "Distillation beats most human writers at scale" | Paying humans to write completions "now just seems absolutely outlandish" |
| **Preference data (RLHF)** | **Mixed.** Academic work shows synthetic does comparably, but frontier labs treat human preference data as a **moat** | "When I talk to industry, it seems like they all use human data in some capacity." Academic work struggled to show human data moved evaluations, which "points to evaluation things" |
| **Evaluation** | LLM-as-a-judge scales the *scoring*, but benchmarks and labels still need human grounding | "Most evaluations that people use to hill climb the models are now based on synthetic data" |

The pattern: **humans set the frontier and seed initial progress; synthetic data scales it.** Around ChatGPT's launch, human data was "a central driver of progress."

---

## Slides: "Model collapse: An outdated worry" (two slides)

**The worry** (Shumailov et al. 2024): repeatedly training a model on its own generations narrows the distribution. The figure (reconstructed from the slide caption) shows the distribution getting narrower over generations, with "the tails go first": rare facts and styles disappear, and small mistakes compound.

**Lambert's take:** this was "a flash in the pan moment," but he had to argue about it "quite regularly." Looking at the abstract today, "I would be so embarrassed as an author of this paper." *(Correction, labelled: he guesses it was "a Nature or a Science article". It was **Nature**, 2024.)*

**Why it doesn't happen in practice:** collapse is mostly a failure of **unfiltered, single-model, self-training loops**. Real pipelines avoid it by:
- mixing in real / human data
- using **diverse teachers**
- **deduplication**
- **strong quality filters**: "you take some of the outputs from the model and you filter them. So you don't train on the things that are really repetitive and are gonna break the model."

Evidence (Gerstgrasser et al. 2024; Feng et al. 2024): synthetic data can, and should, be used at scale without the catastrophic regressions.

*(My note: the collapse setup is "the model trains on its own unfiltered outputs, forever." That sounds like on-policy RL, so why doesn't RL collapse? Because RL doesn't train on its outputs blindly. Each output is **weighted by an advantage** (lecture 3), so bad outputs are pushed *down*. Filtering is a crude version of the same thing: advantage 1 for kept samples, 0 for dropped ones. That's lecture 2's rejection sampling and lecture 5's STaR.)*

---

## Slide: "Canonical, early synthetic datasets and their scale"

| Dataset | Year | Prompts | Tokens (approx.) | Why it mattered |
|---|---|---|---|---|
| **Stanford Alpaca** | 2023 | 52K | ~10M | Early chat SFT (OpenAI outputs) |
| **UltraFeedback** | 2023 | – | – | AI preferences. "Kickstarted the DPO revolution" (`../lecture-6-dpo/`) |
| **Tülu 3** | 2024 | ~1M | ~500M | Pre-RLVR SFT set (Lambert's team) |
| **OpenThoughts 3** | 2025 | 1.2M | ~10B | General reasoning traces (lecture 5 `02`) |

The trend: **~1000× more tokens in two years.** Part of that is many more prompts (synthetic prompt generation creates diversity). Most of it is that reasoning traces are long. *(My note: 10B tokens ÷ 1.2M prompts ≈ 8,000 tokens per example, versus ~200 for Alpaca. That's the `<think>` chain from lecture 5.)*

---

## Slide: "Two meanings of 'distillation'" (key slide)

| | **Technical: knowledge distillation (KD)** | **Colloquial: today's usage** |
|---|---|---|
| Definition | Train a smaller **student** to match a stronger **teacher's full output distribution** | "Train a weaker model on the outputs of a stronger model" |
| Training target | **Soft labels**: the teacher's probability for *every* token in the vocabulary | **Hard text**: the tokens the teacher actually wrote |
| Origin | Hinton, Vinyals & Dean 2015 | Grew out of the technical term |
| Needs | The teacher's **log-probs** (so, its weights, or an API that returns them) | Just the teacher's **text** (an API is enough) |
| Example | Gemma / Gemini Flash, Qwen 3's small models (lecture 5) | "Chinese models distill from GPT/Claude"; R1-distilled models; OpenThoughts |

Lambert: "this will not go away... distillation is now a loaded term." Any model trained on a dataset like the ones above "would be called distilled from the model that generated the completions."

*(A tiny example, my addition. The teacher's next-token probabilities after "The capital of France is" are: " Paris" 0.90, " located" 0.06, " the" 0.04.*
- *Colloquial distillation: the teacher wrote " Paris", so the student's target is " Paris" = 1, everything else 0 (one-hot, like any SFT).*
- *Technical KD: the target is the whole spread 0.90 / 0.06 / 0.04. The student also learns that " located" was a reasonable second choice. That extra information is what Hinton called the teacher's "dark knowledge.")*

### Slide: "Aside: Distillation on Interconnects this year"

Two posts from Lambert's newsletter on the policy fight over distillation:
- *How much does distillation really matter for Chinese LLMs?* (Feb 2026): how much of Chinese open models' gains actually trace to distilling frontier APIs.
- *The distillation panic* (May 2026): why reacting quickly to frontier labs' "fear-based messaging" on distillation could hurt the broader ecosystem.

---

## Slide: "Distillation 1: Classic knowledge distillation"

*(Reconstructed from the slide caption and transcript.)*

```
              same input x
             ┌─────┴─────┐
             ▼           ▼
       TEACHER (big)   STUDENT (small)
             │           │
     soften with τ>1   soften with τ>1
             │           │
   soft distribution q   soft distribution p
             └─────┬─────┘
                   ▼
           loss = KL(q ‖ p)   → update the STUDENT only
```

Both models see the **same input**. The student is trained to match the teacher's whole probability distribution, measured by a **KL divergence** (the distance between two distributions, lecture 3 `02`).

**Temperature τ > 1** "softens" both distributions: it flattens them so the small probabilities become visible. *(My note: with τ = 1, the example above is 0.90 / 0.06 / 0.04. At τ = 2 (divide the logits by 2 before the softmax, which is the same as taking the square root of each probability and renormalizing) it becomes about 0.68 / 0.18 / 0.14. Now the student can actually see, and learn from, the teacher's opinion about the runners-up.)* Hinton's original setting was image classification, so the slide says "relationships between classes."

Lambert's verbal summary of what technical KD needs: "you essentially need to generate the log probabilities of a given token or a completion from a teacher model and the same for the student model. And then you use a KL-based loss to update the student." It's "a more complicated loss function than a lot of things you'll see," because it needs log-probs from **two** models.

---

## Slide: "Distillation 2: The synthetic-data generation pipeline"

*(Reconstructed from the slide caption and transcript.)*

```
prompts ──► STRONG MODEL (teacher) ──► completions ──► (prompt, completion) dataset ──► SFT smaller models
                                         │
                       optional: edit completions / make preference pairs / filter for quality
```

"Yes, this is intentionally very simple as a diagram." This is the colloquial meaning, and it's just SFT: "you take some prompts, you take a teacher model, you generate completions, and then in the future, you train some model on those prompts and completions."

**What changed with reasoning models** (the box on the slide):
- Early on: train on a strong model's API outputs.
- Reasoning models made it harder. You want the **reasoning traces**, and closed APIs stopped returning them. So distillation shifted to:
  - **More aggressive jailbreaking** to recover the hidden traces (controversial)
  - **Open models** more often, because with the weights you get the full trace

**Connects to:** lecture 5 `02`. R1's last stage ("distill into smaller models") and OpenThoughts 3 are both exactly this diagram.

**Where it goes next:** Lambert now turns to technical KD "because I think this is kind of the core math that you'll see again and again when learning post training. ... All the math looks the same. It's all like an expectation of reward, the RL math, or managing KLs." That's `02`.

---

## Summary of this chapter

| Question | Answer |
|---|---|
| What is synthetic data? | Any training data a model helped make: prompts, completions, preferences, rewards, filters |
| Why did it win? | 100–1000× cheaper and much faster; possible once GPT-4-class models arrived |
| Where do humans still matter? | The frontier, preference data (a lab "moat"), grounding evaluations |
| Is model collapse real? | Only for unfiltered self-training loops. Filtering, mixing and diverse teachers prevent it |
| Two meanings of distillation? | **Technical**: match the teacher's full distribution (needs log-probs). **Colloquial**: SFT on the teacher's text |
