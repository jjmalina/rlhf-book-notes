# Introduction & Motivations

Notes on *How Language Models Use Tools and the Path to Agents | Post-Training Course, Lecture 11* (Nathan Lambert), 00:00–07:29. Based on `source/lecture-11-tools-and-agents-slides.md` and `source/lecture-11-tools-and-agents-transcript.md`. Builds on `../lecture-5-reasoning-models/` (RLVR, and the verifier turning into an environment). See `00-map.md` for where tool use sits in post-training and in the training loop.

---

## Where this fits

Lectures 3–5 trained a model that reads a prompt and writes **one** answer, which a reward model or verifier then scores. This lecture is about what happens when the model can **act in the middle of its answer**: run code, search, edit files. That's tool use, and it's the road from "reasoning model" to today's agents (Claude Code, Codex).

Lambert almost skipped this chapter because the book chapter (ch. 13) "is pretty light." He kept it because the basics of tool use are "the foundations" of today's agents. It's also the only lecture built around a single *skill*: "tool use is probably the most lasting skill, and it's almost like a property of models."

The one idea to hold onto:

> **The model still only emits tokens. Tool use means some of those tokens are read by software as an action, the software runs it, and the result is pasted back into the token stream. Everything in this course (SFT, preferences, RL, KL, distillation) is then used to train the model to behave well under that altered token flow.**

---

## Slide: "What are the limitations of model weights alone?"

A section break. Lambert's framing: when ChatGPT came out, you could think of it as "model weights you download from Hugging Face in a vLLM wrapper." Systems have grown more complex since, mostly **to fix the limits of weights alone**.

---

## Slide: "Two asks, two failure modes"

| Ask | Why weights alone fail | What fixes it |
|---|---|---|
| *"Who is the president today?"* | **Knowledge cutoff.** The weights were trained on historical text and "don't know the future." Raw research models (he cites his own Olmo demos) get this wrong | **One search query** |
| *"Move all the arXiv papers in my downloads folder to `~/research/`, named by date."* | The weights **cannot even attempt it.** It needs acting on the world, not describing it. "The model weights only generate tokens, and you need a system that can translate tokens into the real world." | **File-system tools** (or a shell) |

The general point: once "some tokens can be used as actions," a language model can be plugged into almost any tool. (He mentions robotics as the obvious extension, which is out of scope.)

From the slide: tool use "started by addressing structural limitations in LLMs and grew into the central way to push performance." And it's **a trained skill**, so everything in the course applies to it.

---

## Slide: "What an 'LLM' is has changed"

**An LLM today = model weights + tools + harness.**

| Part | What it is | Analogy *(my addition)* |
|---|---|---|
| **Model weights** | The trained network: knowledge, reasoning, style | The person's brain |
| **Tools** | The actions the model can request: search, code execution, file edits, APIs | The person's hands and phone |
| **Harness** | The software loop around the weights that executes those requests and feeds results back into the context (plus the system prompt, permissions, etc.) | The desk, the job description and the rules of the office |

Lambert: post-training is now partly "making sure your model can use tools" and "can work in a harness." This lecture is "a good capstone," moving from the foundations of lectures 1–10 to what we actually use.

His live aside: the slides themselves are made with **Colloquium**, his own slide tool "designed for agents," whose main interface is a **CLI**. "I'm using tools specifically made for language models." In other words, tools are now designed with the LLM as the user.

---

## Slide: "Where we are today"

Tool use started as **"call a calculator"**, used for hard multiple-choice questions and code execution. Now:

- **Coding and terminal agents** (Claude Code, Cursor) drive engineer productivity, using tools that are **general computer primitives** (a shell, files).
- **Deep research, computer use, productivity copilots** use specialized tools for one task. **Deep research came first**: it used search; computer use added browser and navigation tools. These "culminated" in Claude Code and Codex.
- **The hardest current evals are end-to-end tasks in complex containers.**

*(The slide's image, a coding agent in a terminal making many chained tool calls (read files, edit, run tests) within one user turn, isn't in the sources. Reconstructed from the slide caption.)*

Lambert's two most general tools:

| Tool | What it gives the model |
|---|---|
| **Search** | **Getting information** that isn't in the weights |
| **Code execution** (and computer-use tools, closely related) | **Acting**: running a structured process |

"Getting information and acting... these are very complementary to language models that only see tokens."

On evaluation today: "we're putting the model weights with this harness into an environment where tools are defined, and it must act to accomplish a goal." That sentence already contains the three pieces of an agentic RL run: **policy (weights) + harness + environment**. Come back to it in `04`.

---

## Slide: "This lecture"

The plan:

1. **Basics & history** (`02`): why models need tools.
2. **Infra basics** (`03`): interleaved generation, MCP, harnesses, implementation trade-offs.
3. **Training at scale** (`04`): tool-use RL and why it's hard. This part goes beyond the book, using recent reports he reviewed: Nemotron, Kimi K3, GLM-5 and others.

---

## Slide: "Quick pause for YouTube: How'd you end up here?"

He asks viewers to comment whether they're following the whole course or arrived via search. (He says "we're on lecture 10" here; it's lecture 11. *Correction: a slip of the tongue. The title and his opening line both say 11.*)

---

## The whole section in four lines

1. **Weights alone have two gaps:** they don't know anything after their cutoff, and they can't act on the world. Tools close both.
2. **An LLM is now a system: weights + tools + harness.** The harness runs the tool calls and feeds results back.
3. **Search (get information) and code execution (act) are the most general tools.** Coding/terminal agents are the frontier, and the hardest evals run whole tasks in containers.
4. **Tool use is a trained skill**, so every post-training method in the course applies to it. The rest of the lecture is how it works mechanically (`03`) and how it's trained (`04`).
