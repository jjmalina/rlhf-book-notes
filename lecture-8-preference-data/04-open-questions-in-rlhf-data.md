# Open Questions in RLHF Data

Notes on *Preference Data: The Most Opaque Part of Post-Training | Post-Training Course, Lecture 8* (Nathan Lambert), 31:11–35:52. Based on `source/lecture-8-preference-data-slides.md` and `source/lecture-8-preference-data-transcript.md`. Follows on from `03-a-brief-overview-of-preference-data.md`. The last file of the lecture.

---

## Where this fits

The closing section: "things that I think are interesting and why RLHF is an unsolvable problem, which makes for a good place to do something like an academic PhD if you can figure out the storytelling to convince people it is relevant."

The one idea to hold onto:

> **Preference data carries systematic biases (length, lists, flattery, agreeing with the user) straight into the reward model, and RL then amplifies them. Deeper still: nobody can publicly check whether the values written into the labeling spec ever show up in the trained model.**

---

## Slide: "Bias: What to watch for in data"

"Subtle, systematic biases sail straight from the data into the model":

| Bias | What happens | Lambert's comment | *(My addition)* toy example |
|---|---|---|---|
| **Prefix bias** | The **opening** of a response disproportionately drives the label | He wasn't sure of the details on camera ("Oh, I forget"); his reading: "the beginning of the response can bias the label a lot" | "Great question! Here's a clear answer:" wins over a better answer that starts plainly |
| **Sycophancy** | Agreeing with the user over being right | "Well established": telling users what they like to hear makes them pick it | User: "I think 0.1 + 0.2 = 0.3 exactly in floats, right?" The "Yes!" answer gets picked over the correct "Not quite" |
| **Verbosity** | Longer answers rated higher | It "seem[s] more complete and more trustworthy despite the labelers actually not reading it all. So, lazy labelers is definitely a huge and repeated problem" | A 400-word answer beats an equally correct 80-word one |
| **Formatting** | Lists and bold look "better" | "RLHF and preference data is where this markdownification of ChatGPT responses came from. Everything's in lists. Everything's bold." | The same content as bullets with bold headers beats plain paragraphs |
| **Flattery / fluff** | Decorative language inflates scores | "Not new" | "A truly fascinating and profound question..." |

The second half of the slide: **"Detecting and controlling these biases is central to collecting high-quality preference data, and they are exactly what over-optimization amplifies into model behavior"** (`../lecture-9-over-optimization/`).

Lambert: avoiding them all "is kind of an infinite problem because depending on your domain and the strength of the model, there will always be new types of bias."

### When does a bias do its damage? *(my addition)*

The bias enters at labeling time, gets baked into the reward model, and then grows during RL:

```
BEFORE RL
  labeling:        labelers pick the longer answer 60% of the time, even at equal quality
  reward model:    Bradley–Terry learns "length → slightly higher score"     (frozen now)

DURING RL (lecture 3 loop, repeated thousands of times)
  step 2:  policy writes some long and some short completions
  step 3:  frozen reward model scores the long ones a bit higher
  step 5:  long completions get positive advantages
  step 7:  update → the policy writes slightly longer text next batch
  ...      repeat: a small bias in the data becomes a large behavior in the model
```

A 60/40 tilt in labels becomes, after enough steps, a model that always rambles. That's why lecture 3's KL leash exists, and why lecture 4 watches response length (`../lecture-4-implementing-rl/06-loss-aggregation-strategies.md`). With RLVR (lecture 5) this loop is much weaker, because a verifier doesn't care how long or bold the answer is.

---

## Slide: "Higher-level complexities"

Questions about **who** labels and **how**, beyond individual biases:

| Question | What it asks |
|---|---|
| **Collection context** | Do workplace labels transfer to end users? Paid vs. volunteer? Do annotators follow the instructions or their own values? |
| **Type of feedback** | Does a binary pair actually capture the preference we mean? What structure mirrors how people really compare? |
| **Population and demographics** | Who labels? Is disagreement **noise or signal**? |
| **Are the preferences even expressed in the models?** | (Next slide) |

Lambert: "If you're using paid versus unpaid annotators, they will have very different issues." On democracy: "I'm not gonna say that AI models will ever be democratic... They're built by for-profit companies." But "figuring out who is making the [labels] could be really important to figuring out who the biases are, and then changing that who can solve some biases."

*(My note: "noise or signal" links back to the goldfish poems in `01` and Arrow in `02`. If 6 of 10 labelers prefer poem A, Bradley–Terry treats the 4 as noise around a single "true" preference. If the 4 are a distinct group with different taste, averaging them away means the model serves the majority only.)*

---

## Slide: "The unaudited gap: Spec → data → behavior"

"RLHF's **motivation** (align to human preference) has drifted from its **practice** (make models effective)."

The chain that nobody can check from the outside:

```
SPEC                         DATA                          BEHAVIOR
labeler instructions    ──►  preference pairs         ──►  trained model
/ Model Spec                 (biased, `03` and above)      does it actually
"be honest, be concise"                                    act this way?
       │                                                        ▲
       └──────────────── never publicly audited ────────────────┘
```

"Because industrial RLHF is closed, we can't check whether the trained model actually reflects the spec given to annotators." OpenAI's **Model Spec** (2024) documents intended behavior, "but the link from data → behavior stays largely unaudited."

Lambert says this is the question he's most interested in: "you go through all this effort to create a data spec, to collect the data, and to train the model... do the values that people write down in this data collection spec ever actually get reflected in the final model?" "This type of audit was never done publicly." That's why he wants more open datasets like HelpSteer.

His own direction: "longer-term character training work to write a model spec in public, to show how our models fit it or don't, to show what degrades it," as "a living object with full transparency of data, intention, and artifact outcome."

The slide notes that several of these questions already came up with synthetic data (`../lecture-7-synthetic-data/`): **the balance between human and AI feedback**, and **on-policy preference data**.

---

## Slide: "The nature of preferences is the lasting problem of RLHF"

"One of the least-settled, most human parts of the field. Read widely and go to the primary sources. Is a great academic problem!"

Lambert: there are "such fundamental problems to collecting preference data, to training reward models, to integrating it into complex training pipelines." It's good for academics because "it's not too busy, the problems are well motivated, and there's established literature" to build on.

---

## Slides: "The course so far" + closing

| # | Lecture | Book chapters |
|---|---|---|
| 0 | Prerequisites review | |
| 1 | Overview | 1–3 |
| 2 | IFT, reward models & rejection sampling | 4, 5, 9 |
| 3 | RL: Motivation & math | 6 |
| 4 | RL: Implementation & practice | 6 |
| 5 | The rise of reasoning models | 7 |
| 6 | Direct preference optimization | 8 |
| 7 | Synthetic data & modern post-training | 12 |
| **8** | **Preferences & preference data** (today) | 10–11 |
| 9 | Over-optimization & RLHF's bad reputation (next) | 14, app. B |
| 10 | Regularization tools & understanding how post-training changes models | 15 |

Lambert's preview: "when there are imperfections in the data or in the model, you get over-optimization. And regularization is the tool to fix it and to understand it" (`../lecture-9-over-optimization/`, `../lecture-10-regularization/`).

---

## The whole section in four lines

1. **Biases** (prefix, sycophancy, verbosity, formatting, flattery) go from labels into the reward model, and RL **amplifies** them step after step. New ones keep appearing.
2. **Who labels and how** (paid vs. volunteer, demographics, pair vs. richer feedback) shapes the data. Disagreement may be **signal**, not noise.
3. **The unaudited gap:** nobody can publicly check whether the spec given to labelers ends up in the model's behavior.
4. The nature of preferences is RLHF's **lasting, unsolved** problem, and a good one for academics.

---

## The whole of lecture 8 in seven lines

1. **Preference data is the training signal for when there's no right answer.** It rests on "judging is easier than generating."
2. **It's used *before* the RL loop:** pairs → Bradley–Terry reward model (frozen, called at step 3 by PPO or GRPO-with-RM), or pairs → DPO directly. RLVR (lecture 5) skips all of this and judges with a verifier *during* the loop.
3. **One datapoint = (prompt, chosen, rejected)**, plus optional Likert strength, ratings and annotator info, which are mostly metadata.
4. **"Preference as one number" comes from centuries of theory** (Port Royal → Bentham → Ramsey → VNM → Bradley–Terry), whose assumptions (stable, transitive, one-dimensional, aggregatable) don't hold for real labelers (Arrow, Sen, Hirschman, behavioral economics).
5. **Costs ≠ rewards ≠ preferences.** RLHF borrowed RL's optimizers, which assume a fixed closed-form reward, without its guarantees. The reward model is a proxy, and RewardBench accuracy is a proxy for a proxy (objective mismatch).
6. **In practice, preference data is opaque and expensive:** many interfaces, almost all binarized to pairs. Contracts are relationship-driven, ~$500K, with early batches thrown out, and the data must be on-policy. Structured pairs from checkers are a cheap alternative in narrow domains.
7. **Biases in the labels get amplified by optimization** (lecture 9), and nobody can audit whether the spec ever reaches the model. It's RLHF's lasting open problem.
