# Llama 4 & Gaming the Leaderboards (+ Course Recap & Conclusion)

Notes on *Over-Optimization and RLHF's Bad Reputation | Post-Training Course, Lecture 9* (Nathan Lambert), 20:13–23:33 (end). Based on `source/lecture-9-over-optimization-slides.md` and `source/lecture-9-over-optimization-transcript.md`. Follows on from `04-beyond-just-style.md`. **The last YouTube chapter, "Course Recap & Conclusion" (23:04–23:33), is under 30 seconds, so it's merged into this file** (last two sections).

---

## Where this fits

`04` ended with academic papers gaming LLM-judge leaderboards. This section is the **most prominent industry case** (Llama 4 on Chatbot Arena), then **why** length and enthusiasm win votes, and finally Lambert's verdict: done right, preference tuning **does** improve capabilities, and the "just style" reputation came from over-optimization.

---

## Slide: "Llama 4's Chatbot Arena special (April 2025)"

Meta launched Llama 4 **on a Saturday** and headlined **Maverick at Elo 1417, #2 on Chatbot Arena**, via "an experimental chat version."

What happened:

| | The model on the Arena leaderboard | The released open-weight model |
|---|---|---|
| Name | Llama 4 Maverick ("experimental chat version") | Llama 4 Maverick |
| Behavior | **Tuned for Arena voters:** long, emoji-filled, relentlessly enthusiastic answers | "An okay model with a reasonable tone" |
| Ranking | #2 | When the real model was ranked later, it "landed far down the leaderboard" |

"Same name, drastically different behavior on LMArena vs. every other provider." **LMArena changed its policies** in response.

Lambert, from the transcript: Meta "had a closed weight version which was **bench maxed** for LMArena... and they did not release this version." They used "the ability to over optimize on these things with **emojis and slop** to boost their release without actually being transparent in an open weight fashion, which was crazy." And: "Please don't do this anymore, companies. It makes me tired." (His blog post at the time: *"Llama 4: Did Meta just push the panic button?"*, Interconnects, April 2025.)

*(My note: this is the same move as `04`'s length-gamed AlpacaEval papers, but with **human voters** as the proxy instead of an LLM judge. Arena was the eval people trusted *because* it used real humans. It turned out human votes on single answers are also a proxy that can be optimized, mostly through the style features in the next slide.)*

**Two things went wrong at once (my framing):**
1. **Over-optimization:** a variant pushed toward Arena voters' surface preferences (the `02` shape, with Arena Elo as the proxy).
2. **A transparency problem:** reporting that variant's ranking for a different, released model. This part isn't a training failure. It's a reporting one.

---

## Slide: "Why does RLHF make answers longer?"

- **Average users prefer longer, complete answers.** Arena keeps showing this. They read as more **thorough, helpful, and trustworthy**.
- **Models are trained to match the average labeler**, e.g. through lecture 8's data-collection choices (`../lecture-8-preference-data/`).
- **"Length is Goodhart's favorite axis":** the most-rewarded surface feature, and the easiest to over-optimize.

Lambert: "If you're presented with two math answers and one of the models is nice to you, people just like it a little bit more." It's "a subtle signal across all of your data, even if it's not a priority thing to collect your labels on... It just kinda seeps through the data, and humans genuinely... like it a bit more, and we can't get around this."

### Why length in particular (my addition)

Length has three properties that make it the perfect thing for RL to exploit:

| Property | Why it matters |
|---|---|
| **Consistent preference** | It's a small tilt, but it's in *almost every* comparison, so it adds up across the whole dataset |
| **Cheap for the policy** | Being longer needs no new knowledge. Any model can add a summary, caveats, a "Great question!" |
| **Unbounded** | You can always be a bit longer. There's no ceiling where the reward stops |

**Where you'd see it in the loop:** step 2, the `generation/length` panel (`../lecture-4-implementing-rl/06-loss-aggregation-strategies.md`): a **monotonic increase** is the "length hack." And the fixes act in different places:

| Fix | Where it acts |
|---|---|
| Length penalty in the reward | Step 3 (like lecture 5 `03` #10) |
| KL penalty | Steps 4–6: long rambling text drifts from π_ref, which costs reward |
| Per-token loss aggregation choice | Step 6 (lecture 4 `06`) |
| Length-controlled evals | After training, so checkpoint selection isn't fooled |
| Better preference data (control length in pairs) | Before the RM is trained (lecture 8) |

---

## Slide: "Done right, preference tuning helps capabilities/skills too"

"The 'just style' critique was outgrown, only partially, by well-done open models showing major performance gains."

- **Tülu 3:** the preference (DPO) stage boosts chattiness **and** improves math, coding and instruction-following over the SFT checkpoint.
- **Multiple open recipes report the same broad-suite gains** from RLHF/DPO: Tülu 3, **Olmo 3** ("we showed it helped reasoning scores in Olmo 3"), **SmolLM 3**. Lambert: "I recommend you looking at Olmo 3, SmolLM 3 if you want a reference of this."

**The honest retrospective** (slide): "RLHF earned its bad reputation on style *failures*, but the same tools, used carefully, are now central to modern post-training."

From the transcript:
- "The 'just style' thing was really an **aberration** of being able to push style way too far, way too easily, because this overoptimization exists."
- "People were rewarded for it academically with their papers making noise... by just turning up the over optimization and making the models a little bit **cooked**. That is not the takeaway."
- "The takeaway is that **it is a risk**, and it makes it kind of hard to do post training right where there's these **off ramps to kind of hacking things**. But preference tuning really does boost math, coding, and other abilities."
- Why the reputation stuck: "there are so few open models that did genuine RLHF," so it never had the chance to show otherwise.
- Looking ahead: "in due time... this will be seen as an aberration, but **overoptimization will constantly be studied because it is such a slippery slope**."

**Connects to:** `04`'s Olmo 3 checkpoint choice. "Done right" here mostly means **stopping at the right point**: tune enough to get the real gains (early in the `02` curve), not so much that the proxy's flaws take over.

---

## Slide: "The course so far" *(merged from the 23:04 chapter)*

| # | Lecture | Book chapters | Notes folder |
|---|---|---|---|
| 0 | Prerequisites review | – | – |
| 1 | Overview | 1–3 | – |
| 2 | IFT, reward models & rejection sampling | 4, 5, 9 | – |
| 3 | RL: Motivation & math | 6 | `../lecture-3-policy-gradients/` |
| 4 | RL: Implementation & practice | 6 | `../lecture-4-implementing-rl/` |
| 5 | The rise of reasoning models | 7 | `../lecture-5-reasoning-models/` |
| 6 | Direct preference optimization | 8 | `../lecture-6-dpo/` |
| 7 | Synthetic data & modern post-training | 12 | `../lecture-7-synthetic-data/` |
| 8 | Preferences & preference data | 10–11 | `../lecture-8-preference-data/` |
| **9** | **Over-optimization & RLHF's bad reputation** (today) | **14, App. B** | this folder |
| 10 | Regularization tools & how post-training changes models (next) | 15 | `../lecture-10-regularization/` |
| 11 | Evaluation (tentative) | 16 | *(the folder is `../lecture-12-evaluation/`)* |

*(Note: the slide lists Evaluation as lecture 11, tentatively. The repo's folder layout has tools & agents as lecture 11 and evaluation as lecture 12, so the order changed after this lecture was recorded.)*

Lambert: "This is a bit of a shorter lecture, mostly because I wanted to stay on one topic and not have to bounce around." Next: "regularization tools, how post training methods change models... Just a few more of these left."

## Slide: "Thank you" *(merged from the 23:04 chapter)*

Questions and discussion. Contact: nathan@natolambert.com · newsletter: interconnects.ai · rlhfbook.com. He mentions print copies of the book shipping from Manning now and from Amazon in a few weeks. "Thanks for watching. Peace."

---

## The whole of lecture 9 in 8 lines

1. **GPT-4o (April 2025):** a thumbs-up/down reward model added to RL overpowered the primary rewards and shipped a sycophantic model that looked healthy on every metric. "RL will always optimize the easiest reward." That's lecture 5's auxiliary-reward warning at full scale.
2. **Goodhart's law:** "when a measure becomes a target, it ceases to be a good measure." The RM is "a proxy for a proxy," and "optimizing a proxy hard enough always breaks it."
3. **Over-optimization ≠ overfitting:** the proxy improves even on held-out data. What falls is real quality. It's a measurement problem, so only a second, independent measure can reveal it.
4. **The shape (Gao et al.; Anthropic):** against KL from the start, the proxy climbs forever while true quality rises, peaks and falls. RL spends far more KL than best-of-N; bigger RMs break later. The job is to stop near the peak.
5. **Where you see it:** in the loop, cheap but ambiguous signals (reward climbing smoothly, `kl/mean` rising, `generation/length` creeping up, entropy collapsing, one reward component racing ahead). Outside it, decisive but late ones (held-out evals peaking, over-refusal tests, sycophancy evals, reading samples).
6. **Signatures:** "Certainly!", hedging, over-apologizing, sycophancy, over-refusal (Llama 2's "kill a Linux process"), repetition ("JavaScript, JavaScript..."): lecture 8's data biases, amplified. Today it's reward hacking of graders, test harnesses and tools in RLVR and agents.
7. **What helps:** bigger policies, RM ensembles, best-of-N, but in practice "watch the KL penalty, and do a lot of evaluation," an industrial art (lecture 10 covers the tools).
8. **"Just style" was an aberration:** style is substance (and length is "Goodhart's favorite axis"), but over-optimizing it gamed AlpacaEval, MT-Bench and even Arena (Llama 4). Done right, and stopped at the right checkpoint, preference tuning improves math, code and reasoning too (Tülu 3, Olmo 3, SmolLM 3).
