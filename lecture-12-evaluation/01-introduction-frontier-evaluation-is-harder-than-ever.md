# Introduction: Frontier Evaluation Is Harder Than Ever

Notes on *How Evaluation Has Evolved with Frontier Model Progress | Post-Training Course, Lecture 12* (Nathan Lambert), 00:00–03:39. Based on `source/lecture-12-evaluation-slides.md` and `source/lecture-12-evaluation-transcript.md`. Builds on `../lecture-3-policy-gradients/00-training-loop-timeline.md` (the loop that evals sit outside of) and `../lecture-5-reasoning-models/` (RLVR, pass@k). See `00-map.md` for the "when does evaluation happen" diagram and the eras timeline.

---

## Where this fits

Lectures 3–11 were about **making models better**. Lecture 12 is about **how you know they got better**. This corresponds to chapter 16 of the RLHF book plus appendix C (practical eval variance).

There's very little math here. It's mostly history, plus a set of practical warnings about why a single benchmark number can mislead you.

Lambert calls it "one of my favorite chapters in the book. It is succinct. It tells a story."

---

## When does this happen?

The timing question for this lecture has a simple answer: **evaluation is outside the training loop.**

```
inner loop (steps 1–7, lecture 3)       ← lectures 3–5. Changes weights every step.
   │
   │ every N steps: save a checkpoint
   ▼
EVALUATION                               ← this lecture. Reads a checkpoint, produces a number.
   │                                       Changes nothing by itself.
   ▼
a person decides: keep this data mix? ship this checkpoint? change the reward?
```

So "evaluation" is the measuring stick for every decision *between* training runs. The step-3 reward is a measuring stick *inside* one run. They can use the same checker and still be different things (full table in `00-map.md`).

---

## Slide: "How we measure progress changes over time"

The core thesis:

- **Different model types need different strategies.** GPT-3 (a base model), ChatGPT, o1/o3 and today's agents each had "behavioral traits that need different types of evaluation strategies."
- **As models get smarter, we need to re-invent the cutting edge of AI.** Lambert: "What we are doing as a field is collectively reinventing the cutting edge of AI, and evaluation is the core piece of this."
- **Evaluation is how we know what the models really are.**

The incentive argument for why this matters to a post-training course:

> "Post-training researchers are super good at making numbers go up on evaluations. But if we don't have the evaluations, it's going to be hard to do this."

*(My note: this is the RL story one level up. In lectures 3–5 the **policy** maximizes a **reward**. Here the **researchers** maximize an **eval**. Both optimizers find whatever the measuring stick rewards, whether or not it's what you meant. That's why lectures on reward hacking (lecture 3 `02`, `../lecture-9-over-optimization/`) and this lecture's Part 3 on eval gaming are really the same idea.)*

---

## Slide: "Frontier evaluation is harder than it ever has been"

When **Claude Opus 4.6** and **GPT-5.3-Codex** shipped in the same week (February 2026), the headline benchmark differences were tiny "and settled nothing about which model to use."

What actually separated them was found **through use**: usability, product fit, and behavior over long agentic tasks. Lambert's Interconnects post on this is called *the post-benchmark era*.

How his view has moved over time:

| Earlier complaint | Current complaint |
|---|---|
| "We can't trust the evaluations of frontier models because they can **cherry-pick** which evaluations they use" | The evaluations **aren't capturing what matters to the user**: "this trust, this robustness, how it works in products" |

He thinks the importance of evaluation is "undersold."

*(My note, for someone building a coding agent: this is the gap between "model X scores 2 points higher on SWE-bench Verified" and "model X, inside my harness, on my repos, finishes more tickets without me having to clean up." The second is what you care about, and only your own eval measures it. Part 3 (`04`) ends with exactly this advice: "run your own evals.")*

---

## Slide: "...and the tasks we're trying to measure keep taking longer"

*(The METR plot isn't in the folder. This is reconstructed from the slide caption and transcript.)*

METR's **time horizon** measure: take tasks, measure how long they take a human expert, and find the task length a model completes **50% of the time**. Plotted against model release date (on a log scale), it makes a roughly straight line. According to the slide, it **doubles roughly every 7 months**, going from seconds-long questions to tasks that take experts hours (*Time Horizon 1.1*, January 2026).

The consequence for evaluation:

> "Measuring the frontier now means running hours-long expert tasks, many times over."

Lambert adds that METR is "reaching the limit" of the task data they originally used to make this plot. Then the line that sums up the whole problem:

> "For areas where we can measure performance... **you can hill climb faster than you can make the target.** And making the targets is really, really hard, and that's what evaluation researchers do."

*(My note: a small worked example of why longer tasks are a cost problem. A 30-second quiz question, run 10 times for a stable average, is 5 minutes of model time. A 4-hour task, run 5 times on each of 100 tasks, is 2,000 hours of agent time, each hour holding a sandbox and paying for inference. This is where the ">$100K" figure for a full agentic suite at the end of the lecture comes from.)*

---

## Slide: "This lecture"

> "Every training decision in this course — data mixes, hyperparameters, which checkpoint ships — is made off benchmark numbers."

The plan:

| Part | Topic | File |
|---|---|---|
| 1 | **The eras**: how each model type was prompted, graded and benchmarked | `02` |
| 2 | **Agentic evals**: the system around today's scores | `03` |
| 3 | **Trusting the number**: variance, contamination and gaming | `04` |

---

## The whole section in four lines

1. **Evaluation sits outside the training loop.** It reads checkpoints and produces numbers that people use to make every training decision (data mix, hyperparameters, which checkpoint ships).
2. **How we measure must keep changing**, because each kind of model (base → chat → reasoning → agent) needs a different kind of test.
3. **The frontier is now hard to measure:** benchmark gaps between top models are tiny and say little, and the real differences show up in long, real use.
4. **Tasks keep getting longer** (METR: the 50%-success task length doubles about every 7 months), so evals are getting slower and more expensive. Researchers "hill climb faster than you can make the target."
