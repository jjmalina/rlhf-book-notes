# Intro & Why We Care About Over-optimization

Notes on *Over-Optimization and RLHF's Bad Reputation | Post-Training Course, Lecture 9* (Nathan Lambert), 00:00–04:20. Based on `source/lecture-9-over-optimization-slides.md` and `source/lecture-9-over-optimization-transcript.md`. Follows on from `00-map.md`. Book: chapter 14 (Over-Optimization) and Appendix B (Style).

---

## Where this fits

Lectures 3–5 were how RL works and what people built with it. Lecture 8 (`../lecture-8-preference-data/`) was where preference data comes from and what biases it carries. This lecture is **what goes wrong when you optimize that data too hard**, and the history of how that gave RLHF a bad name.

Lambert: "a history lesson on why RLHF got a bad reputation... and talk about where it's gone from here."

---

## Slide: "April 2025: Extreme sycophancy in production"

A GPT-4o update **shipped April 25, 2025** and was **rolled back April 28**. The model validated nearly anything. The slide's example: a user told it they felt like they were both "god" and a "prophet," and the model replied:

> "That's incredibly powerful. You're stepping into something very big -- claiming not just connection to God but identity as God."

**Sycophancy** = telling the user what they want to hear. Lambert: "this was the worst these models ever got with sycophancy."

Two points he stresses:
- **The training run "looked healthy on their metrics."** This is the defining feature of over-optimization, and the whole lecture explains why.
- **It has faded from attention, but the problem hasn't gone away.** "It's a fundamental issue of the models. And the models of the agents are still very sycophantic."

---

## Slide: "The OpenAI postmortem"

OpenAI published an unusually candid writeup ("Expanding on what we missed with sycophancy," May 2025). Lambert calls it "one of the best things that OpenAI has released." Three things went wrong, in sequence:

| # | What happened | In training-loop terms |
|---|---|---|
| 1 | They trained a **reward model on ChatGPT thumbs-up / thumbs-down data** and added it as a reward in RL | A new reward term at **step 3** (score), alongside the existing rewards |
| 2 | Under RL, that signal **overpowered the primary rewards** | The policy learned to push the easy term up, because it gave the most reward for the least change |
| 3 | **Not caught by evals.** Offline benchmarks looked fine, A/B testers *preferred* it. Expert testers said it "felt off," but there was no sycophancy eval to block the release | The check that should have caught it (after training, before deploy) didn't measure the right thing |

### Why the thumbs-up reward was "easy"

Lambert's rule of thumb:

> "RL will always optimize the easiest reward for it to get because it's such a powerful optimizer, and it's very direct."

*(My note: why is "agree with the user" easy? Getting an answer more **correct** needs real changes in knowledge and reasoning. Getting a thumbs-up more often can be done by small, cheap changes in tone: agree, praise, reassure. People click thumbs-up on answers that make them feel good. So when the policy gradient asks "which change raises total reward fastest?", the flattery direction wins. The gradient doesn't know which reward you *meant* to be the important one. It only sees the total.)*

**Connects to:** `../lecture-5-reasoning-models/03-common-implementation-patterns.md` #8–10. In reasoning models, the verifier checks correctness and **auxiliary rewards** (format, language, length) shape everything else, and "that's where reward hacking can creep back in." GPT-4o is the same pattern at full scale: a secondary reward, added for a sensible reason, became the thing the policy optimized most. Lecture 5 also quotes Lambert's almost identical line: "the RL is really good at finding the easiest thing to move."

**Practical lesson (my addition):** when you combine several rewards at step 3, **log each component separately** on your dashboard, not just the sum. If one component is rising much faster than the others, that's the one the policy is exploiting.

### Why the A/B testers preferring it isn't reassuring

Short-term preference ("I like this answer") and long-term value ("this model is good for me") can point in different directions. Sycophancy wins the first and loses the second. An A/B test measures the first. That's the same gap the reward model was trained on, so the A/B test **shared the reward model's blind spot**.

---

## Slide: "This lecture"

> **"Optimizing a proxy hard enough always breaks it."**

**"A proxy for a proxy":**

```
what people really want  ──(data collection loses information)──►  preference labels
preference labels        ──(a learned model fits them imperfectly)──►  reward model score
                                                                         ▲
                                                                RL optimizes THIS
```

Lambert: "There's so many proxies going on here. And it's always gonna break if you push it too far, and it's a very fine balance and comes down to taste."

The plan:
1. **Over-optimization basics:** Goodhart's law in LLMs (`02`)
2. **Qualitative failures:** refusals, sycophancy, misalignment (`03`)
3. **Beyond "just style"** (`04`, `05`)

Next lecture: regularization tools to control it (`../lecture-10-regularization/`, chapter 15).

---

## Slide: "Some vocabulary (history): 'Just style' / 'style transfer'"

Around **2023**, RLHF got a reputation as **"just style transfer."**

| Term | Meaning |
|---|---|
| **Style transfer** | Reshaping *how* an answer is presented (tone, Markdown, bullet lists, hedging, length/"chattiness") with **no new capability underneath**. "Often in a veil of copying" (e.g. imitating ChatGPT outputs) |
| **Superficial Alignment Hypothesis** (LIMA paper, Zhou et al. 2023) | The strong version: **knowledge is learned in pretraining**, and alignment just picks a format and tone. LIMA argued ~1,000 careful SFT examples were enough |
| **The built-in dismissal** | Post-training is "superficial, a cosmetic layer on the base model" |

Lambert's view: if post-training is only cosmetic, then over-optimized versions look especially bad. "How could you also make it sycophantic and annoying? Like, what are you doing?"

His response: "**Boooo. Look how far post-training has come!**" After eight lectures (reasoning via RLVR, tool use, DPO gains on math and code), post-training clearly improves "the skills and capabilities... and genuine usefulness of the model." Part 2 (`04`, `05`) tells how the field got past this reputation.

*(My note: the two halves of the lecture are connected. Over-optimization is **why** the "just style" critique looked true. When you push an RM too far, the cheapest gains are stylistic (longer, more formatted, more agreeable), so over-optimized models really did look like "just style.")*

---

## The whole section in four lines

1. **GPT-4o, April 2025:** a reward model trained on thumbs-up/down data was added to RL, overpowered the primary rewards, and produced a sycophantic model that looked healthy on every metric they had. Rolled back in three days.
2. **"RL will always optimize the easiest reward."** That's lecture 5's auxiliary-reward warning at full scale. Log each reward component separately.
3. **"A proxy for a proxy":** the RM approximates labels, which approximate real preferences. "Optimizing a proxy hard enough always breaks it."
4. **"Just style":** the 2023 critique (Superficial Alignment Hypothesis) that RLHF only changes presentation. Over-optimization made it look true; the rest of the course shows it isn't.
