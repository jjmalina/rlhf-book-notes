# Part 2: An Intro to Agentic Evals

Notes on *How Evaluation Has Evolved with Frontier Model Progress | Post-Training Course, Lecture 12* (Nathan Lambert), 17:36–21:19. Based on `source/lecture-12-evaluation-slides.md` and `source/lecture-12-evaluation-transcript.md`. Follows on from `02-the-eras-of-post-training-evaluation.md`. Builds on `../lecture-11-tools-and-agents/` (harnesses and tool calls). Slide credit for this part: Florian Brand.

---

## Where this fits

In `02`, the pipeline was **prompt → model → grader**, and the prompt was where most things went wrong. In the agent era, **the model is one box among many**, and every box can change the score.

The point to hold onto:

> **Two labs running "the same benchmark" can measure meaningfully different things.** Most of the differences are undocumented implementation choices, and most of them only make scores *worse*.

Lambert: "You normally get performance degradations from little things going wrong rather than things being way better."

**When does this happen?** Still after training, on a frozen checkpoint. But one "eval sample" is now a whole **multi-turn episode**: the model calls tools, reads their output, and calls more tools, possibly for hours. Then the final state of the sandbox is graded.

*(My note: an agentic eval episode is structurally the same thing as a **multi-turn RL rollout** (step 2 of the lecture 3 loop, in its agentic form from lecture 5 and `../lecture-11-tools-and-agents/`), followed by the step-3 check. The difference is what happens to the result: in training it becomes an advantage and a gradient; in an eval it becomes one entry in an average. That's why, in `04`, training and eval converge on "the same environment code.")*

---

## Slide: "The agentic pipeline: the model is one piece"

*(The diagram isn't in the folder. This is reconstructed from the slide text and transcript.)*

```
 ┌───────────────────────── HARDWARE (GPU type, batch size, tensor-parallel setting) ──────────────┐
 │                                                                                                 │
 │   task prompt                                                                                    │
 │       │                                                                                          │
 │       ▼                                                                                          │
 │   ┌─────────── HARNESS ───────────┐    tool calls     ┌──────────── SANDBOX ────────────┐        │
 │   │ system prompt, tool defs,     │ ───────────────►  │ container on cloud CPUs:        │        │
 │   │ context management,           │                   │ repo files, shell, test runner, │        │
 │   │ the agent loop                │ ◄───────────────  │ network rules                   │        │
 │   │        │  ▲                   │   tool outputs    └────────────────┬────────────────┘        │
 │   │        ▼  │                   │                                    │                         │
 │   │  INFERENCE ENGINE / API       │                         final state / trajectory             │
 │   │  (vLLM, SGLang, in-house)     │                                    │                         │
 │   │  runs the MODEL, parses       │                                    ▼                         │
 │   │  tool calls                   │                     GRADER: regex, tests, or LLM judge       │
 │   └───────────────────────────────┘                                    │                         │
 │                                                    TIMEOUT ─── cuts off the episode if too long  │
 └────────────────────────────────────────────────────────────────────────┼─────────────────────────┘
                                                                          ▼
                                                                        score
```

The definitions from the slide and transcript:

| Box | What it is |
|---|---|
| **Harness** | "The software that manages the model, the prompts, the context management, the tools it has access to." The agent loop |
| **Sandbox** | "A reproducible world with the files, tools and rules of the task." Lambert: "CPUs with files and tools on them that the model has to interact with." Run in the cloud |
| **Engine or API** | What actually runs the model: vLLM, SGLang, or an internal engine. "Can have a big impact" |
| **Hardware and timeouts** | Which GPUs, which settings, and how long each task is allowed to run |
| **Grader** | Regex or an LLM judge, over trajectories that may be hours long |

---

## Slide: "Harnesses are key"

- **Frontier models are trained in their own harness.** Evaluating them in a different one **under-reports** capability. Lambert: OpenAI and Google models "are trained to work in their own harness from the start." **Open models are trained to work in every harness**, which is "a very different thing." Recent releases from Kimi and others advertise being **harness-agnostic**.
- **The extreme case**, from the ARC-AGI-3 technical report: on one environment, **Opus 4.6 scores 0% with no harness and 97.1% with a hand-crafted one.**
- So **"same model, different agent product" produces wildly different scores.**

Lambert's prediction: we used to have prompts specialized to one domain, and now "a lot of startups have harnesses specialized to one domain as their way of showing success." That builds real economic value ("they genuinely can be" the best), but makes standards "much messier to establish."

**Concrete harness differences for a coding agent (my addition).** Same model weights, same SWE-bench-style task. Things your harness decides that change the score:

| Harness choice | How it can sink the score |
|---|---|
| **Edit tool format** | The model was trained with a `str_replace(old, new)` tool. Your harness only offers "write the whole file" or a unified-diff tool. It makes more malformed edits, or rewrites 2,000-line files and truncates them |
| **Tool output truncation** | Your harness cuts tool output to 2,000 characters. The failing test's traceback is at the *end* of a long pytest log, so the model never sees why it failed |
| **Context management** | On long tasks, the harness drops or summarizes old turns. A bad summarizer throws away the file paths the model already found, so it searches again and times out |
| **System prompt** | No instruction to run the tests before finishing, so the model submits untested patches. The trained harness's prompt always said to |
| **Max turns / stop condition** | A 30-turn cap on tasks that typically take 60 |

None of these change the model. All of them change the number. If you're comparing models for your own agent, either use each model's native harness (and say so) or use one fixed harness for all (and say so). Those are two different questions: "which product is best" versus "which model is best inside my product."

---

## Slide: "Many little things contribute to agentic scores"

> "Every implementation detail in the boxes on [the pipeline diagram] is a knob someone chose (usually undocumented)."

### 1. The inference engine can be different

vLLM's public **postmortem on serving Kimi K2**: three engine bugs held **tool-call success below 20%**. After the fixes: **99.9%**. **Same weights.**

Lambert: the bugs caused "misformatting... making the model just not able to call tools." He trusts vLLM "a lot because they will be transparent about it," and says "so many players have had issues with tool use." If you call the same model across providers on OpenRouter, "there could be so many issues."

*(My note on what this kind of bug looks like: the model emits its tool call in its trained format (special tokens + JSON). The engine's **tool-call parser** has to turn that text into a structured call. If the parser expects slightly different delimiters, or mishandles a model-specific token, the call is either dropped or passed as plain text. The model "said" the right thing, but the harness never ran it. This is the agentic version of the answer-extraction bug from `02`. It's also cousin to lecture 5's FP32 LM-head fix: the inference stack and the training stack have to agree exactly on what the model produced.)*

**How to catch it (my addition):** log the **tool-call parse-failure rate** separately from task success. A healthy setup is near 0%. If one provider shows 15%, the score you're seeing is a serving bug, not the model.

### 2. Hardware-level differences

- **Variance across GPUs is real** "and a headache." Lambert: "you realize that you need to run all of your evals on Hopper GPUs with a specific batch size because the kernels and FlashAttention are slightly different."
- Some benchmarks measure hardware on purpose (KernelBench-style tasks need specific GPUs).
- **With scaled sandboxes, one bad GPU or machine can stall the whole system and tank evals.**

*(My note on why numerics change scores: changing the batch size or the number of GPUs a model is split across changes the **order** in which numbers are added. Floating-point addition isn't exactly associative, so the logits differ in the last few digits. Usually that doesn't matter. But at a token where two choices are nearly tied, it flips which one wins, and in a 10,000-token reasoning trace or a 50-turn agent episode, one early flip sends the rest down a different path. Same weights, same prompt, same seed, different answer.)*

### 3. Timeouts are crucial on challenging tasks

Models often time out on hard tasks. **Rerunning Terminal-Bench 2 with 3–5× longer timeouts moved GPT-5.2 by +6 to +15 points.**

Lambert: "You're gonna have these straggler tasks that take a really long time. And if those were gonna end up being right, you're gonna be suppressing your score by not running your timeouts long enough."

**Worked example (my addition).** A 100-task suite, 30-minute timeout:

| | Tasks |
|---|---|
| Solved in under 30 min | 50 |
| Would have been solved in 30–90 min | 10 |
| Never solved | 40 |

With the 30-minute timeout the score is **50%**. With a 90-minute timeout it's **60%**. The model didn't change; the budget did. And a slower inference provider (fewer tokens per second) uses up the same wall-clock timeout with less thinking, so **provider speed leaks into the score** too.

Practical rules for a dev-agent eval:
- Set timeouts **per task**, generously (some test suites alone take 10 minutes).
- Report **timeouts as a separate outcome** from failures, so you can see when the budget is the bottleneck.
- Measure in **tokens or turns**, not just wall-clock, if you want a number that doesn't depend on server load.

---

## The whole section in five lines

1. **In agentic evals the model is one box.** The score depends on the **harness** (prompts, tools, context management), the **sandbox** (the task's world), the **inference engine**, the **hardware**, the **timeout** and the **grader**.
2. **Harnesses matter most.** Frontier models are trained in their own harness, so a different one under-reports them. Extreme case: Opus 4.6 on one ARC-AGI-3 environment, **0% with no harness and 97.1% with a custom one**.
3. **Engine bugs can masquerade as model weakness:** Kimi K2 on vLLM went from **<20% to 99.9%** tool-call success after three bug fixes, with the same weights.
4. **Hardware and numerics** (GPU type, batch size, tensor-parallel setting) shift scores, and one bad machine can stall a scaled eval.
5. **Timeouts suppress scores on hard tasks** (Terminal-Bench 2: +6 to +15 points with 3–5× longer timeouts). Almost every one of these knobs is undocumented, and they mostly push scores *down*.
