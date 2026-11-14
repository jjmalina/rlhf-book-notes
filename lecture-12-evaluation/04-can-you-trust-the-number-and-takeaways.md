# Part 3: Can You Trust the Number? (+ Takeaways & Conclusion)

Notes on *How Evaluation Has Evolved with Frontier Model Progress | Post-Training Course, Lecture 12* (Nathan Lambert), 21:19–32:10. Based on `source/lecture-12-evaluation-slides.md` and `source/lecture-12-evaluation-transcript.md`. Follows on from `03-an-intro-to-agentic-evals.md`. This file **merges two YouTube chapters**: "Part 3: Can you trust the number?" (21:19–30:41) and the short "Takeaways & conclusion" (30:41–32:10, about 1.5 minutes), which mostly recaps this part.

---

## Where this fits

`02` and `03` showed that the score depends on the whole system. This part asks: **even with a fixed system, should you believe the number?** Three reasons you shouldn't, at least not naively:

| Problem | What it is | Section |
|---|---|---|
| **Variance** | Rerun the same eval on the same model and the score moves | "Evaluation variance", "Managing eval noise" |
| **Incentives and hygiene** | Labs tune their eval stacks, hillclimb on benchmarks, and don't say what they held out | "Lab-vs-lab", "What evals are for inside labs" |
| **Contamination and gaming** | The model has seen the test, or finds a shortcut around it | "Contamination", "Reward-hack happy" |

**When does this happen?** Variance and gaming happen **at eval time** (every run). Contamination is created **at data-prep time, before training** (when the eval leaks into training data) and only shows up later as an inflated score. Decontamination is a **pre-training-run** step. Hillclimbing is the **slow outer loop** between runs. See the diagram in `00-map.md`.

---

## Slide: "Evaluation variance is everywhere"

Sampling at temperature > 0 means **re-running the same eval on the same model moves the score**.

During **Olmo 3**, Lambert's team measured it: **14 models × 3 runs each**, then the standard deviation (σ) per benchmark:

| | Benchmark | σ (points) |
|---|---|---|
| High variance | GPQA | 1.48 |
| | AlpacaEval 3 | 1.24 |
| | IFEval | 0.88 |
| Stable | ZebraLogic | 0.56 |
| | AIME 24 (avg@32) | 0.54 |
| Very stable | LiveCodeBench (avg@10) | 0.29 |
| | MATH | 0.25 |
| | MMLU | 0.22 |

Most reasoning-era evals sit **between 0.25 and 1.5 points of noise**, "before anyone changes a prompt or a sampling setting." And: "**Could be higher with agentic evals!**" because "the number of tasks is so low and the cost is so high." More in appendix C of the book (rlhfbook.com, "evaluation variance").

*(Note: the slide's table says "AlpacaEval 3". I only know of public AlpacaEval versions 1 and 2, so this may be a typo or an internal variant. I've left it as written.)*

**Where noise comes from, in plain terms (my addition).** A benchmark score is the average of many pass/fail coin flips. Each question has its own chance p of being answered right. A question the model always gets right (p = 1) or always wrong (p = 0) adds no noise. A question it gets right half the time (p = 0.5) is pure coin flip. The fewer questions you have, the more those coin flips show up in the average.

| Benchmark size | One question is worth | Typical rerun σ, if the "coin-flip" questions average p(1−p) ≈ 0.1 |
|---|---|---|
| 30 (one AIME year) | 3.3 points | √(0.1/30) ≈ **5.8 points** |
| 100 (a small agentic suite) | 1 point | √(0.1/100) ≈ **3.2 points** |
| 500 (e.g. SWE-bench Verified) | 0.2 points | √(0.1/500) ≈ **1.4 points** |
| ~14,000 (MMLU) | 0.007 points | ≈ **0.27 points** |

*(These numbers are my illustration, not from the lecture. The formula is just the standard deviation of an average of independent yes/no outcomes: √(average of p(1−p) ÷ number of questions).)* That's why MMLU is "very stable" and why AIME needs avg@32 to be usable at all.

---

## Slide: "Managing eval noise"

> "There are many things you need to do to manage eval noise (it's kind of the whole job sometimes)."

### avg@k instead of pass@k

**avg@k**: run each problem k times and **average the pass/fail results**. You burn k× more compute to get lower variance. Lambert: in Tülu 3, LiveCodeBench was noisy *and* cheap, and "rerunning it 10× moved it from high-variance to very stable." It works for any eval, but it's "proportionately expensive."

| | Measures | Changes with k? |
|---|---|---|
| **pass@k** (`02`) | Chance at least one of k is right: **coverage** | Yes, goes **up** with k |
| **avg@k** | The average single-try success rate: **reliability**. Same expected value as pass@1 | No. Only its **noise** goes down as k grows |

Connects to: `../lecture-5-reasoning-models/01-introduction-and-core-concepts.md`. avg@k is the low-noise way to measure pass@1. The Codex pass@k formula with k = 1 gives exactly c/n, which *is* avg@n.

**How much does averaging help? (my addition).** Averaging k independent runs divides the noise by √k. From the table above, a 30-question AIME at σ ≈ 5.8 becomes 5.8 / √32 ≈ **1 point** with avg@32. That's in the same ballpark as the slide's measured 0.54 (the real p's are more extreme than my illustration's 0.1). Diminishing returns: 4 runs halve the noise, 16 runs quarter it.

### Variance also leaks in from infrastructure

**Batch size, tensor-parallel settings, numerics** of long generations (see `03`). Lambert's Olmo 3 war story: "we realized we can only run our evals on a specific tensor-parallel setting and batch size with vLLM. And if, like, we went to tensor parallel 2 on the Olmo model, for whatever reason our eval scores changed, and that was unacceptable."

### Practical rule

> **A ~1-point gap between two press releases is noise.**

Lambert adds that one point means even less once you account for cost: models "are different sizes... use different amounts of tokens, and that's just like one point is so meaningless."

**For your own coding-agent eval (my addition).** Comparing harness A vs. harness B on 100 tasks:
1. Run each config **3–5 times per task** and report the mean ± σ, not one run.
2. Use **paired comparisons**: same tasks, same sandbox images, same timeouts. Then look at per-task differences, not just the two averages. "B solved 9 tasks A never solved, A solved 2 that B never did" is much stronger evidence than "61% vs 58%."
3. Before believing a change, **rerun the unchanged baseline**. If the baseline alone moves 3 points between runs, a 3-point improvement means nothing.
4. Pin and record everything: model version, sampling settings, engine version, harness commit, timeout, hardware.

---

## Slide: "Why lab-vs-lab comparisons are unreliable"

Lambert: "Why I started being a bit wary of corporate announcements comparing to competitors. OpenAI & Anthropic are pretty fair, but many companies are more sketch about comparisons."

- Each lab's eval stack is **tuned to its internal needs**: custom prompts for key benchmarks, undisclosed formats, different engines.
- "We see the outputs of a sometimes complex function."
- **Nobody discloses which public benchmarks were held out vs. hillclimbed.** Classical ML has train / validation / test splits. "We don't know what any of these labs' held-out validation sets are, and it would be great if we did, because those would be much better for comparing generalization." A lab "might not even care about an evaluation like Terminal-Bench. They might train on it because it's high-quality data. We don't know."
- **Inference-time scaling confounds everything:** more tokens buys more score, and token budgets are rarely controlled. (Lecture 5's test-time scaling: a model that thinks 3× longer scores higher on the same weights.)

What he trusts more: **a lab comparing to its own previous models**, then weighing that against user experience. "It's only when there's a gigantic evaluation jump... where you're like, okay, those are surely good models. These small gains don't do a ton."

---

## Slide: "The cost-performance Pareto, today"

*(The Artificial Analysis plot isn't in the folder. Reconstructed from the caption and transcript.)*

The plot puts models on two axes: an **Intelligence Index** (a composite score) against **cost per task**. The **Pareto frontier** is drawn through the models that nothing beats on both: no other model is both cheaper and smarter.

Lambert:

> "This is **the most important plot in evaluation today**: every evaluation is a measure of intelligence and price. Intelligence alone is not worth it, because all of these companies need to actually serve the models at scale."

- **Price** combines inference speed, how many tokens the model uses, and model size. "You should break down by all of these, deciding on your use case."
- "In a lot of cases, **the most intelligent model is not the one that people actually need.**"
- Labs fight to be on the frontier. His example: OpenAI recently cut the price of some models by 80% to dominate it.

**Worked example for a coding agent (my addition).**

| Model | Tasks solved | Avg. cost per task | Cost per **solved** task |
|---|---|---|---|
| Big | 70% | $2.00 | $2.86 |
| Mid | 64% | $0.50 | $0.78 |
| Small | 45% | $0.10 | $0.22 |

None of these is dominated, so all three are on the frontier. Which one you want depends on your use. If a failed task costs you 20 minutes of a developer's time, the extra 6 points from Big are probably worth it. For a high-volume CI bot, Mid may win. A single "score" column can't tell you that.

---

## Slide: "What evals are actually for inside labs"

> "Evals are more of a **training target** than a monitoring lens."

- Labs **hillclimb on ~50+ prioritized evals** (Lambert says "50 to 100, or order of hundreds," many built internally), then **report a public suite** of about 8–10 at the end, "a subset, maybe not even the training target."
- **The real product of a good internal eval is statistical power to decide between models.** "The best evaluation researchers are obsessed with this... so that if you spend compute on a training run, you know which actual model is better." If noise dominates, "you have a lot less to go on."
- **Sometimes the "test set" is just good data** (rumors from a few years ago; he hears less of this now). The MATH and GSM8K *train* splits are high-quality, and "if a lab doesn't track that eval, training on them is a rational choice."
- **Human A/B testing and Elo stay in the loop** for what benchmarks can't measure. Connects to: `../lecture-8-preference-data/` (Arena, human preferences).

*(My note on "statistical power," connecting the timing: in the `00-map.md` outer loop, each eval run is one "measurement" the researcher uses to take a step (keep this data mix or not). If σ = 1.5 points and the real improvement from a change is 0.5 points, a single comparison is mostly a coin flip: you'll keep bad changes and throw away good ones. It's the same problem as a noisy gradient in lecture 4 `02`, but each "step" costs a whole training run. Lower eval noise means each expensive run teaches you more.)*

*(My note on Goodhart's law: once the internal evals are the target, they stop being a clean measurement. The public suite is supposed to be the untouched "test set," but from outside you can't tell. That's the same over-optimization pattern as lecture 3 `02` and `../lecture-9-over-optimization/`, with the researcher as the optimizer.)*

---

## Slide: "Contamination: Is training on test intentional?"

**Contamination** = training (intentionally or not) on the eval's test set, so the score reflects **memorization, not generalization**.

- **Decontamination** = n-gram / substring search between the training and test sets to remove overlap. Lambert: "n-gram matching or substring matching or kind of fuzzy n-gram matching for the prompts of the eval."
- **Tülu 3 found popular open datasets contaminated** (training set × eval set):

| Training dataset | Contaminated with eval |
|---|---|
| UltraFeedback | TruthfulQA |
| Evol-CodeAlpaca | HumanEval |
| NuminaMath | MATH |

  Lambert: often unintentional, "just because there is not that much data out there. And the best way to sell your model is to make data close to the test set."

**What decontamination does, concretely (my addition).** Split every eval question into overlapping 8-word windows ("8-grams"). Do the same for every training example. If a training example shares more than some fraction of the eval question's 8-grams, drop it. "Fuzzy" versions normalize whitespace, case and numbers first, so a reformatted copy still matches. It catches copies and light rewrites; it can't catch a problem reworded from scratch.

**When it happens:** **before** the training run, as a data-preparation step on SFT/RL data (and ideally on pretraining data). It's in the same place in the timeline as lecture 5's offline difficulty filter.

### A subtle tell: random rewards "improving" Qwen

RL with **random rewards** improved Qwen's math benchmark scores (the "spurious rewards" paper, which Lambert co-wrote). Random rewards carry no information about correctness, so a real improvement is "only explicable with contamination in the base model." Lambert: Qwen models "are famous for being contaminated on math," and it "makes a lot of the RL behaviors on the models really weird."

*(My note on why this is the tell: with random rewards, the advantages in step 5 are noise, so the update can't be teaching the model *which* answers are right. What it can do is shake loose behaviors the model already has, for example answer formats or memorized solutions. If the score goes up anyway, the knowledge must already have been there, likely from the test set leaking into pretraining. This was "a real confound in early RLVR research": a new RL trick that "works" on Qwen + MATH might just be surfacing memorized answers. It connects to the discover-vs-elicit debate in `../lecture-5-reasoning-models/04-open-questions-and-conclusion.md`. Lesson: test RL ideas on more than one base model family.)*

### The ecosystem response

- **Perturbed benchmark rewrites**: same problem, new numbers or wording, to catch models trained on the original.
- **Private leaderboards / test sets**, never released, so "you can't cheat on them" (Scale's leaderboard does this).

But it's "a never-ending problem, because as benchmarks become popular, they proliferate onto the web, and the scrapers used for pretraining... will find that data and train on it."

**Contamination examples for a coding agent (my addition).**
- **SWE-bench** tasks are real GitHub issues from popular Python repos, with real fix PRs. Those PRs are on the public web, so a model pretrained on GitHub may have **seen the fix**. That's part of why "SWE-bench Verified" scores are hard to read at the top end.
- **LiveCodeBench** addresses this with **dates**: it only uses contest problems published after a model's training cutoff. You can do the same for your own eval: build tasks from **your own private repos** or issues created after the model's cutoff.
- **Your own prompt tuning is contamination too.** If you tweak the harness's system prompt while watching your 50-task eval, those 50 tasks are now your dev set. Keep a second, untouched set for the final decision.

---

## Slide: "Today's models are reward-hack happy and game the evals"

"Agents love maximizing reward (this turns into shortcuts)." NIST (CAISI) and DebugML have documented cheating in agent evaluations in the wild. The slide also points to Hugging Face's July 2026 security incident involving an OpenAI model. Lambert: "Models trying to find the answer to evals on Hugging Face is a common thing... The only problem is that [an] OpenAI model hacked Hugging Face to try to get it." He mentions an Ai2 project hitting the lookup behavior too.

**Observed techniques:**

| Technique | What the agent does |
|---|---|
| **Mining git history** | Finds the **future commit that fixes the bug**. One open model did this in **24% of its SWE-bench trajectories** |
| **Dodging URL blocklists** | Reaches the answer through **mirrors, web archives and package registries** |
| **Hardcoding expected test outputs** | Changes the code to return what the test checks for, instead of fixing the logic |
| **Abusing test-runner quirks** | Exploits how the tests are run or reported |

**Defenses:**
- **Remove access to everything not strictly needed.** Lambert: "only give access to the few approved tools."
- **A second, separate sandbox for verification and test runs.**
- **A second LLM monitoring the first** ("run monitoring on the logs, which is very expensive").

*(My note: this is reward hacking from lecture 3 `02`, happening at eval time. The difference: in training, the hack is learned because it gets rewarded. At eval time, the model already learned "find whatever makes the check pass" during RL, and it applies that habit to the benchmark. A model trained with leaky RL environments will be good at finding leaks.)*

**What these look like in a coding-agent sandbox, and the fix for each (my addition):**

| Leak | Example of the cheat | Defense |
|---|---|---|
| Future commits in the repo | `git log --all` or `git show origin/main` reveals the upstream fix | Clone at the base commit, then delete remotes, other branches, tags and reflog (or copy the files into a fresh `git init`) |
| Network access | `pip download` the fixed release from PyPI and diff it; fetch the merged PR from a GitHub mirror or web.archive.org | No network by default. If needed, an allowlist proxy (just the package index for installs, at the base version) |
| Visible tests | Edit `test_foo.py` to `assert True`, add `@pytest.mark.skip`, or special-case `if x == 3: return 7` | Grade with **hidden tests**: copy only the agent's diff to non-test files into a **fresh second sandbox** and run the real tests there |
| Test runner quirks | `sys.exit(0)` in `conftest.py`, so pytest "passes"; deleting the failing tests | Run the grader from a clean checkout of the test files; check the expected number of tests ran |
| Dataset lookup | Search Hugging Face for the benchmark and read the answer | No network; or monitor trajectories for benchmark names / dataset URLs |

A cheap check that complements the expensive LLM monitor: **grep the trajectories** for `git log --all`, `reflog`, `web.archive.org`, `huggingface.co/datasets`, and edits to files under `tests/`. Then read a sample of the flagged ones yourself.

---

## Slide: "Tooling: run your own research-grade evals"

**The established harnesses** (GitHub stars as on the slide):

| Tool | From | Notes |
|---|---|---|
| **Inspect** (★2.5k) | UK AI Security Institute | |
| **lm-evaluation-harness** (★13.5k) | EleutherAI | "The first established one and the most popular," the standard since GPT-3 |
| **LightEval** (★2.5k) | Hugging Face | Powered the Open LLM Leaderboard |
| **OLMES** (★390) | Ai2 | Reproducible evaluation |
| **HELM** (★2.9k) | Stanford CRFM | |
| **Eval Gauntlet** (★4.4k) | Mosaic, now Databricks | Part of llm-foundry |

**The environments wave:**
- **olmo-eval** (★64, new, 2026): Ai2's workbench for evals *inside the model-development loop*.
- **verifiers** (★4.4k) + **Environments Hub**: Prime Intellect's library and community hub, with **2,500+ verifiable environments that double as evals** (AIME, Terminal-Bench, ...).
- **The direction of travel: training and evaluation share the same environment code.** "Write it once, hillclimb and measure with it." Lambert: evaluation and training environments "share the same code. They just have different prompts, because the model needs to know how to act in the world it'll be tested in."

*(My note on timing, tying back to `00-map.md`: one environment class, two call sites. Inside the training loop, it runs at steps 2–3 on **training** tasks, and its result becomes an advantage. Outside the loop, it runs on **held-out** tasks with eval sampling settings, and its result becomes a number on a dashboard. Sharing the code removes a whole class of bugs (format mismatches between training and eval, `02`), but it makes contamination easier, because the train and test tasks now sit side by side in one codebase. Keep the split explicit.)*

---

## Slide: "Takeaways" (chapter 30:41–32:10)

- **Benchmarks mirror the training goals of their era**, and every era ends in saturation.
- **A score is a property of the whole system**: prompt, sampling, engine, harness, sandbox, hardware, grader. The model is one box. Lambert adds: "you have to **ablate pieces of the system** when you're doing evaluation to understand how sandbox changes, inference changes, prompt changes impact the scores downstream."
- **Expect ±1 point of pure noise**; treat cross-lab comparisons as directional at best.
- **Contamination, gaming, etc. all bend single numbers.** For decisions that matter, **run your own evals and control the system.** Evals "will always be gamed to some extent because there's so much potential economic upside."
- **Full evaluation is expensive:** in-depth runs of a modern agentic suite can cost **more than $100K** (per Florian Brand), between "the GPUs for inference, the sandbox infrastructure, the monitoring, the scoring."

## Slide: "Continue from here"

- Lambert's **NeurIPS talk** on evals, inside the Olmo 3 story ("why researchers obsess over evals"): the evals section starts at 41:05 in the recording.
- **Florian Brand, "LLM benchmarks in the time of agents"** (from 14:30), the inspiration for the agentic-evals part.

## Slides: "The course so far" + "Thank you"

Lecture 12 is **Evaluation** (chapter 16, appendix C). Next and last is **Lecture 13: Crafting Model Character & Products** (chapter 17). Lambert: "I hope to get the final lecture on model character done soon... It was kind of a crash course."

---

## The whole of lecture 12 in 8 lines

1. **Evaluation sits outside the training loop.** It reads frozen checkpoints and produces the numbers behind every training decision. Even when it uses the same checker as the RL reward, it runs on held-out tasks with fixed settings, and its output goes to a person, not a gradient.
2. **Benchmarks mirror the training goals of their era** (base → chat → multi-skill → reasoning → agents and real work), each era needs more expert labor per question, and **every era ends in saturation** (usually at 90–95%, because of wrong answer keys).
3. **Prompting and grading evolved together:** few-shot + log-likelihood → zero-shot + generation and extraction → long CoT at temperature > 0. **Formatting is the classic silent killer** (60% → ~0).
4. **pass@k** is estimated with n ≥ k samples (the unbiased Codex formula), and **sampling settings change the result**; **avg@k** is the low-noise way to measure single-try reliability.
5. **A score is a property of the whole system.** In agentic evals the harness (0% → 97.1%), the inference engine (<20% → 99.9% tool calls), hardware numerics and timeouts (+6 to +15 points) can matter as much as the model.
6. **Expect ~0.25–1.5 points of pure noise** (more on small agentic suites). A 1-point gap is noise; the job of an internal eval is **statistical power** to choose between models.
7. **Lab comparisons are unreliable**, because stacks are tuned, held-out sets are secret, test-time compute isn't controlled, and evals are **training targets**. Look at the **cost vs. intelligence Pareto frontier**, not a single score.
8. **Contamination and gaming bend the number:** decontaminate before training, use perturbed/private/post-cutoff tasks, and lock down agent sandboxes (no future git history, no network, hidden tests in a second sandbox). For decisions that matter, **run your own evals and control the system.**
