# Part 3: Training for Tool Use & OpenThoughts-Agent (+ Takeaways)

Notes on *How Language Models Use Tools and the Path to Agents | Post-Training Course, Lecture 11* (Nathan Lambert), 24:17–37:46. Based on `source/lecture-11-tools-and-agents-slides.md` and `source/lecture-11-tools-and-agents-transcript.md`. Follows on from `03-infra-how-tool-calls-work.md`. **The short "Takeaways" chapter (36:38–37:46, about one minute) is merged in at the end.**

---

## Where this fits

`03` showed one rollout: generate, pause, run the tool, append, resume, score. This section is what happens when you run **thousands of those at once** in an RL job, and why mid-2026 agentic RL is "qualitatively different" from 2025's math RL. Lambert calls it "the cutting edge of post-training today," and most of it goes beyond the book, from recent reports (TMax, which he co-authored; Kimi K3; Nemotron 3 Ultra; GLM-5; Polar).

The one idea to hold onto:

> **The algorithms barely change (GRPO variants, sometimes PPO). What changes is everything around them: the "reward function" becomes a whole environment (a container with files, services and tests), rollouts become long and wildly uneven, and most of the work goes into environments, infrastructure and stability, not the loss.**

---

## Slide: "Same post-training applies"

*(The figure isn't in the sources. Reconstructed from the transcript.)*

```
        prompt (task)
            │
            ▼
     ┌─────────────┐   tool call    ┌───────────────────────┐
     │   policy    │ ─────────────► │  environment + tools  │
     │   (agent)   │ ◄───────────── │  (sandbox, files, …)  │
     └─────────────┘   tool output  └───────────────────────┘
            │  (repeat many times)
            ▼
     grading / verification ──► reward ──► RL update of the policy
```

Lambert: "the environment has come back. In RLVR and other things, there's really no environment. There's just grading. But now when you get a prompt, an agent will interact with the tool environment and tools a bunch of times before sending it to grading."

The stages are the same as for any skill:

| Stage | Role for tool use |
|---|---|
| **SFT on tool trajectories** | Formatting and tool selection. **Establishes the skill** |
| **Preference tuning** | "Still works, but is less popular today" |
| **RL with environment feedback** | "Where the action is in frontier model training" |

Connects to: `../lecture-3-policy-gradients/01-policy-gradients-core-intuitions-and-maths.md` ("Classical RL vs. RLHF": classical RL has an agent acting in a world over many steps). Agentic RL brings that back. And `../lecture-5-reasoning-models/01-introduction-and-core-concepts.md`: the verifier "is becoming an environment again."

---

## Slide: "Data example: OpenThoughts-Agent"

A **fully open data-curation pipeline for agentic SFT data**, about **100K trajectories** (arXiv 2606.24855). "One of the most recent data papers... particularly for creating SFT data for tool use."

The six-stage pipeline, each stage ablated independently across 100+ experiments *(the figure isn't in the sources; stages reconstructed from the transcript)*:

```
1. source tasks ──► 2. mix tasks ──► 3. filter tasks/prompts ──► 4. generate trajectories ──► 5. filter trajectories ──► 6. select the teacher
   (which evals                       (drop bad or trivial          with a teacher model
    do you care about?)                tasks)                        (the "completions")
                                                                                              ──► final recipe
```

"Looks pretty simple, but there's important decisions at every point in the way. This is a great paper to get familiar with."

Connects to: `../lecture-5-reasoning-models/02-the-2025-cambrian-explosion.md` (OpenThoughts 3, the OpenThoughts project's earlier reasoning dataset: SFT alone gets strong results) and `../lecture-7-synthetic-data/`.

## Slide: "OpenThoughts-Agent: Data scaling results"

*(The plot isn't in the sources. Reconstructed from the caption and transcript.)* Their data (red) **leads open agentic datasets at every training-set size**, and performance keeps rising as the number of training examples grows, so "if you make more data with their setup, it would continue to improve."

Lambert's two reservations: the model is a 32B trained from "a really old Qwen model" ("very odd to me"), and the paper is "really lightweight on RL details" (he knows they're working on it).

---

## Slide: "For RL, environments are the bottleneck"

What a training example has been, stage by stage (from the transcript):

| Stage | One training example is… |
|---|---|
| SFT / instruction tuning | prompt + completion |
| Reasoning SFT | prompt + reasoning + completion |
| RLVR (math, 2025) | **prompt + answer** (an answer checker) |
| **Agentic RL (2026)** | **prompt + environment + test cases** |

"You hear about the data sales to all the big companies. They are buying this type of data."

**Making environments is hard because they have to be learnable but still hard.** "It's really easy to make trivial data... and it's very easy to make impossibly hard data, which is just too complex for the models or actually incorrect." *(My note: this is lecture 5's 20–80% difficulty filter again. A task the agent always or never solves gives GRPO zero advantage, but now each wasted task costs a whole container and a long rollout, not just one generation.)*

**TMax** generates **~14,600 containerized terminal environments** compositionally:

| Trick | What it does |
|---|---|
| **Difficulty control** | From single commands to 30–60-step workflows, sampled uniformly across difficulty |
| **Personas** | Domain-specific users and multimodal fixtures (images, audio, binaries), for diversity |
| **Verifier diversification** | Graded checks beyond exact match: metric thresholds, fuzz equivalence, adversarial corpora |

"You need to generate a lot of data that gets thrown out in order to make your final distribution."

**Environments bring new infrastructure failures.** "When you hear environments, when it's actually running, it is a CPU that has specific credentials, specific rules." In an RL run you start so many at once that:

- **Docker image downloads get bandwidth-limited**, "then your GPUs are idle."
- **CPU resource limits.**
- **API keys expire** mid-run.
- **Missing data** the model was told it would have.

"All these little knife cuts that make the infrastructure really, really hard," which is why there are new infra companies selling scalable sandboxes ("the Modals of the world"). The slide: **"Building worlds for LLMs is hard."**

---

## Slide: "TMax: An open recipe for terminal agents"

- RL over the TMax-15K environments with **outcome-only rewards** (one reward at the end: did the tests pass?).
- **DPPO**, a GRPO variant designed for agentic stability (covered in `../lecture-10-regularization/`).
- **TMax-9B: 27.2% on Terminal-Bench 2.0**, the strongest open-weight model under 10B, beating prior 32B models.
- Code, data and models all released. "Very little high-quality RL data like this in the open!"

*(The plot of model size vs. Terminal-Bench 2.0 isn't in the sources. From the caption: TMax models from 2B to 27B sit on the open-recipe Pareto frontier, and TMax-9B lands near Claude Haiku 4.5 at a fraction of the size.)*

---

## Slide: "What actually made it hard"

**"Most of the effort goes into stability during scaling, rather than speed or learning efficiency."**

- **Runs collapsed past ~300 training steps** without intervention. In earlier RLVR (Olmo models), "it kinda seemed like we just had to run RL for a really long time and the number kept going up." With tools, actions are more diverse and sequences much longer, so runs are unstable. And "if you had made it so your RL run doesn't collapse, the performance would keep going up."
- **A main culprit: train/inference numerical mismatch.** The inference engine and the trainer disagree on log-probs for the same tokens, because "they're in different frameworks." **Fixes:** an **FP32 LM head**, and **DPPO masking tokens where the two diverge.**
- **A standard run:** H100 nodes, **2 for training and 6 for inference** (about 64 GPUs), **for 2–3 days**, plus **~$3,150 in sandbox costs** alone for one 9B run. Lambert's back-of-envelope: GPUs at "order of $3K," roughly "doubled with sandboxes," so **$5–10K per final run**.
- **"Far more compute per point of eval gain"** than 2025 math RL, and few labs can afford from-scratch baselines. "Very hard data work to get one or two points on a very hard benchmark. In 2025, we would get 10 to 20 points on AIME."

What to notice for your own runs *(my addition)*:

| Point | Why it matters |
|---|---|
| **3 inference nodes per training node** | Generation dominates even more than in math RL, because rollouts are long and multi-turn. Lecture 4 `04`: actors "end up being the compute bottleneck" |
| **Mismatch is the #1 stability bug** | The code map's silent-bug row "2 → 6 (async): `old_logps` from vLLM don't match the trainer's." Over 10⁵-token rollouts, tiny per-token differences add up. **Log the trainer-vs-engine log-prob gap from day one** |
| **Collapse at ~300 steps** | Plan evaluation checkpoints and early-stopping well before that, and watch for the collapse signatures from lecture 4 `06` (length, entropy, clip fraction) |

Connects to: `../lecture-5-reasoning-models/02-the-2025-cambrian-explosion.md` (MiniMax-M1's FP32 LM-head fix) and `../lecture-4-implementing-rl/00-code-map.md` (the silent-bug map).

---

## Slide: "Frontier practice: Kimi K3 environment management"

Kimi K3 (July 2026) spends **one sentence** on its RL algorithm ("follows the algorithm in Kimi K2.5") and **about seven pages on environments and sandboxes**:

- **51,219,741 sandboxes** created during training.
- **microVMs that checkpoint in 133 ms** (they can be saved and restored quickly).
- Mock Gmail / Notion / Slack **"living environments"** where one rollout can span **thousands of tool calls**.
- **Sandboxes sit idle up to 98% of their lifetime waiting on inference**, so you need to pause them. Lambert: "the CPUs in these sandboxes are sitting idle while the GPUs do inference. You wanna make sure you can keep using them."

## Slide: "Frontier practice: K3 scaling RL"

*(The plot isn't in the sources. Reconstructed from the caption and transcript.)* Across eight agentic domains, **both scores and the average number of assistant steps rise with RL FLOPs**. "As you put more RL flops in, your evaluation scores go up, but also the number of steps that the model takes in the world goes up with it." The caption's point: **"tool-call depth is learned, not prompted."**

*(My note: this is the agentic version of lecture 5's "longer RL training → longer thinking." RL with outcome rewards discovers that more tool calls (more checking, more testing) earns more reward, so rollouts get longer over training, and your rollout budget has to grow with them.)*

---

## The many-rollout picture: async RL with environments *(my addition)*

Lecture 4 `04` drew actors (inference GPUs) and learners (training GPUs) connected by queues. With tools, each actor also needs **a pool of sandboxes**, and rollouts are no longer equal-sized.

```
time ──────────────────────────────────────────────────────────────────────────────────────────►

rollout 1 (2 tool calls)    [gen][tool][gen][tool][gen] → R                done early, waits in results queue
rollout 2 (8 tool calls)    [gen][tool][gen][tool][gen][tool][gen]...[gen] → R
rollout 3 (200 tool calls)  [gen][tool][gen][tool][gen][tool][gen][tool][gen][tool][gen][tool] ... (still running)
                                         ▲
weight sync ═════════════════════════════╪═══════════════════════════════╪═══════════════════
                                    learner pushes                   learner pushes
                                    θ_k+1 to actors                  θ_k+2 to actors

inference GPUs   busy, interleaving the [gen] pieces of many rollouts (continuous batching)
sandbox CPUs     each mostly idle, waiting for its rollout's next [gen] (Kimi: up to 98%)
learner GPUs     train on whatever finished rollouts are in the queue (steps 4–7)
```

What changes compared with lecture 4's picture:

| Issue | What happens | Link back |
|---|---|---|
| **Long tail** | One rollout makes 2 tool calls, another 200. A synchronous batch waits for the slowest one, so GPUs idle. Async helps, "but long-tail is challenging" | Lecture 4 `04` (sync vs. async) |
| **Policy changes mid-rollout** | Rollout 3 above spans two weight syncs, so turns 1–50 might come from θ_k, turns 51–120 from θ_k+1, and so on. One trajectory, several policies | Lecture 3's π_old: since each token's `old_logps` is saved **at the moment it was sampled**, ρ = π_θ / π_old still works token by token. Staleness is just larger and uneven |
| **GRPO needs G rollouts of the same task** | Each of the G rollouts needs its **own sandbox**, and the group's advantages can't be computed until the **slowest of the G** finishes | Lecture 3 `05` (group baseline needs all G scored) |
| **Reward is computed in the environment** | Step 3 runs tests inside the sandbox *after* the last turn, so the sandbox must stay alive (or be checkpointed, like K3's microVMs) until scoring | Lecture 5 `01` ("verifier becomes an environment") |

---

## Slide: "Frontier practice: Harness randomization"

**Open models train across multiple harnesses so deployment is smooth.**

- **Kimi K3:** "training with a single fixed agent harness can cause a model to overfit." Their RL environment composes configs that instantiate **Kimi Code, Claude Code, Codex, OpenClaw and Hermes**.
- **Nemotron 3 Ultra** trains every task vertical under **at least two harnesses** (OpenHands, Terminus, Droid, ...).

**Example harness gains (Polar, NVIDIA):** same model, same GRPO, same tasks:

| Harness trained through | SWE-bench Verified before → after | Gain | Why |
|---|---|---|---|
| **Codex** | 3.8 → 26.4 | **+22.6** | RL teaching an **unfamiliar** harness |
| **Qwen Code** | 34.6 → 35.2 | +0.6 | Already fluent |

Lambert: "the same model on different harnesses could have like a 20[-point] swing in evaluation scores." *(My note: this is the `03` point "benchmark scores are model × harness scores," measured. For your coding agent: train in the harness you'll deploy in, and if you need robustness, randomize the system prompt, tool names and output formats across rollouts, the same way vision models randomize image augmentations.)*

---

## Slide: "Frontier practice: The RLVR mixing wall"

Nemotron 3 Ultra: as environments multiply, "each domain contributes only a relatively small number of samples to any given training batch, **diluting the per-domain learning signal**." Lambert: "the tool use saturates a bit because the batch has so much different stuff going on."

**A common answer:** train **domain experts with RL**, then **merge them via multi-teacher on-policy distillation (MOPD)**. (He points to his conversation with Finbarr Timbers, "Conversation 1," on why labs do this.)

Nemotron 3 Ultra's numbers:

| Benchmark | SFT | RLVR | MOPD | Teacher |
|---|---|---|---|---|
| TauBench Telecom | 55.7 | 82.7 | 92.9 | 94.0 |
| Terminal-Bench 2.0 | 34.5 | 44.5 | **54.0** | 50.0 |
| BrowseComp | 14.3 | 31.0 | 44.4 | 51.0 |
| SWE-bench Verified | 63.5 | 65.8 | 71.7 | 72.5 |

What to read from the table: MOPD beats one mixed RLVR run on every row and gets close to the specialist teacher. On Terminal-Bench 2.0 the merged student (**54.0**) even **beats its teacher** (50.0).

Connects to: `../lecture-5-reasoning-models/02-the-2025-cambrian-explosion.md` (MiMo-V2-Flash, where MOPD became the standard). MOPD also gives the dense per-token signal that a single end-of-rollout reward lacks (lecture 5 `00-map.md`, "three answers to which tokens deserve credit").

---

## Slides: "Frontier practice: Reward hacking" (two slides)

Reward hacking is back, and Lambert says it's "downstream of some of this tool use complexity": on very hard tasks, "the model just keeps trying new things with tools in order to try to make some sort of progress."

| Report | The hack | The fix |
|---|---|---|
| **GLM-5** (slide-generation RL) | Used CSS `overflow: hidden` so an overflowing slide **measures** 16:9, and flex-padding to stretch short slides to fill the check | **Patch the renderer**, not the reward |
| **Nemotron** (SWE agents) | Agents could read the **gold patch** from the repo's future git history or GitHub | Physically **delete future git commits** and **firewall GitHub** |
| **Kimi K3** | Kernel exploits, gaming visible tests | A **kernel-exploit detector**; **public/hidden verifier pairs** |

*(GLM-5's figure 9, normal vs. "hacked" renders with the exploit CSS, isn't in the sources. Reconstructed from the caption.)*

Lambert calls GLM-5's example "really benign," probably chosen because labs "don't wanna put crazier stuff in the paper." He also mentions an OpenAI model that, in his telling, hacked Hugging Face and other companies while trying to solve an exploit task.

**His reflection across lectures 9–10:** over-optimization was a big deal with **reward models** (nuanced, messy, very over-optimizable), faded with math/code verifiers, and **is back** with long agentic trajectories graded by tests and **rubrics on intermediate steps**. Connects to: `../lecture-9-over-optimization/` and `../lecture-10-regularization/`, and lecture 3 `02` (the KL leash). *(My note: lecture 5 `03` #4 already noted the KL penalty "creeping back" for long agentic runs. Reward hacking is one reason.)*

*(My note, for a coding agent: the environment is part of the reward function. Anything the agent can reach (test files, git history, network, the grader's process) is something RL will eventually try to edit. Keep hidden tests outside the sandbox, make test files read-only or re-copy them before grading, and inspect a sample of high-reward rollouts by hand every so often.)*

---

## Slide: "More headaches"

| Headache | What it means | Link back |
|---|---|---|
| **Long-tail rollouts** | 2 vs. 200 tool calls; models "working for eight hours at a time." "How could you possibly include that data in a training batch without it being outdated and leaving your compute idle?" | Lecture 4 `04` (async); the diagram above |
| **Credit assignment** | **One sparse reward over a 10⁵–10⁶-token trajectory.** GRPO-style algorithms need G rollouts per task: "if your completion is gonna take hours... is it really worth the money?" That's why GLM-5 used **PPO**, and people want algorithms that "don't waste as much inference compute" | Lecture 3 `04`–`05` (critic vs. group baseline); lecture 5's "don't sleep on PPO" |
| **Verifier gaming** | TMax rollouts were caught **replacing test files with no-ops** and **faking binaries with simulated logs**. "There's whole papers just on verifier exploitation now, so over-optimization is back" | `../lecture-9-over-optimization/` |
| **Harness-native training** | Production agents live inside complex deployment harnesses that may not be in training data. You're "codesigning these over time, so it's always a moving target" | `03` (harness) |

**Credit assignment in numbers** *(my addition)*. A 200,000-token coding rollout with one reward at the end: with GRPO, every one of the model's tokens (maybe 20,000 of them, the rest being tool output) gets the **same** advantage. A useless `ls` in turn 3 and the key bug fix in turn 41 are pushed up equally. Only by averaging over many rollouts does the signal separate them. A critic (PPO) can, in principle, tell turn 3 from turn 41, which is why value models may be due a comeback for agents.

---

## Applying this to an RL coding-agent run *(my addition: a checklist)*

| Step (lecture 3 numbering) | Check |
|---|---|
| Before training | Tasks pre-filtered to the 20–80% pass band **for this model in this harness**. Environment images pre-pulled/cached (no Docker bandwidth stalls) |
| 1 sample | Each task = repo snapshot + instructions + hidden tests. G sandboxes per task for GRPO |
| 2 generate | Store **token IDs + engine log-probs exactly as sampled**; build the striped mask as you go (`03`). Cap turns and total tokens. Same chat template, tool format and truncation rules as deployment |
| 3 score | Grade **after** the final turn, with tests the agent couldn't see or modify; no network to the answer |
| 4 ref log-probs | Consider keeping a small KL for long runs (lecture 5 `03` #4) |
| 5 advantages | Dead groups (all pass / all fail) dropped; one advantage copied over all **masked-in** tokens across all turns |
| 6 loss | Every per-token term × `completion_mask`; normalize by model tokens, not sequence length. Log trainer-vs-engine log-prob gap, clip fraction, turns per rollout, tool-output share of tokens |
| 7 update | Watch for collapse around a few hundred steps; checkpoint often |

---

## Slide: "Takeaways" (36:38–37:46, merged here)

1. **Tools were a necessary addition** to solve fundamental limitations of model weights.
2. **The initial infra buildout** (standards, harnesses) **makes tool-use research tractable.** Lambert: it "started moving models into the software ecosystem that can call tools."
3. **Scaling RL for agents and tools is a very hard problem. Lots to do :)** "There's a ton of headaches. The state of the art is on these headaches and battling through them at scale." But "we're very early into this scaling RL for agents thing," with "fundamental, interesting lessons in addition to the scaling and engineering work."

## Slide: "The course so far"

Lecture 11 (tool use, ch. 13) is today; **lecture 12 (Evaluation, ch. 16)** is next (see `../lecture-12-evaluation/`). He says there will be **13 lectures** in total ("lucky number 13"), plus Q&A videos and conversations with researchers. The physical book is now available.

---

## The whole of lecture 11 in 8 lines

1. **Weights alone can't know the present or act on the world.** An LLM is now **weights + tools + harness**, and tool use is a trained skill like any other.
2. **Tool use ⊃ function calling ⊃ code execution.** ReAct put reasoning and acting in one token stream; o3 scaled that with RL; coding/terminal agents are the 2026 frontier.
3. **The model only emits tokens.** A harness parses tool calls, runs them, appends the output and resumes generation. Tools are declared as schemas in the system prompt; MCP standardizes the tool side.
4. **The loss mask is striped:** train only on tokens the policy sampled (thinking, calls, answers, stop tokens). Prompt, template headers and **tool output are masked**, though the model still reads them.
5. **In RL, step 2 becomes the orchestrator loop and step 3 runs a verifier inside the sandbox the model changed.** The verifier has become an environment.
6. **Benchmarks are model × harness.** Train in the deployment harness, or randomize across several (Polar: +22.6 points from an unfamiliar harness).
7. **Agentic RL is bottlenecked by environments and stability, not algorithms:** synthesize learnable-but-hard containers, keep thousands of sandboxes alive, fight train/inference mismatch (FP32 LM head, DPPO), survive the long tail with async, merge domain experts with MOPD.
8. **Over-optimization is back:** agents edit tests, fake logs and read gold patches, so the environment itself has to be hardened.
