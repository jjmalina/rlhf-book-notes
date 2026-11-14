# Part 2: Beyond "Just Style"

Notes on *Over-Optimization and RLHF's Bad Reputation | Post-Training Course, Lecture 9* (Nathan Lambert), 14:47–20:13. Based on `source/lecture-9-over-optimization-slides.md` and `source/lecture-9-over-optimization-transcript.md`. Follows on from `03-signatures-of-over-optimization-and-misalignment.md`.

---

## Where this fits

Part 1 was the science of over-optimization. Part 2 is **history**: how the field came to see that RLHF (and preference tuning generally) is "more than just style," and why it took so long. The short answer: **style really is valuable, but style is also the easiest thing to over-optimize**, and many 2023–24 papers did exactly that to post big numbers on easy-to-game evals.

Lambert: "it took a long time for the literature to grow out of certain failure modes that were kind of trapping it in its scope of impact."

---

## Slide: "Same base model: what did preference tuning change?"

Lambert's **library** tool, released with the book (rlhfbook.com/library), shows **SFT → DPO** pairs from the same base model on **16 shared prompts**: Tülu 3 70B and OLMo 2 32B / 7B.

| | SFT answer | DPO answer (after preference tuning) |
|---|---|---|
| Shape | "A single wall of prose." Lambert: "very generic, almost like a wall of text" | Definition first, then headers and numbered lists |
| Content | the facts | **the same facts** |
| Extras | – | Markdown, styling, "some clever turns of phrase" |

"It is just very consistent that the RLHF model is just, like, slightly cleaned up."

**Timing reminder (my addition):** the comparison is between two checkpoints of one run: the SFT checkpoint (which also becomes π_ref for the RL or DPO stage) and the model after preference tuning. Since the base and the SFT data are identical, **every difference you see came from the preference stage**.

---

## Slide: "Style as substance"

"An early phase of RLHF's history was convincing people that style was actually useful sometimes, and **not just brainrot**."

Three arguments:

1. **Style is intertwined with what the information is.** Lambert's example is the book *Sapiens*: "all of the history in Sapiens was known. But if you repackage information in a more accessible way, the user gets **meaningful value** and not just entertainment."
2. **Good preference tuning moves real numbers.** **Llama 3 Instruct**'s Arena standing "is widely attributed to its personality: more succinct and clever than other models of its era." (Slide aside: "This is funny, given Llama 4, RIP." See `05`.)
3. **If RLHF makes models more fun to use, that's delivered value**, independent of benchmark scores.

The catch, which is the whole lecture: "well done preference tuning genuinely improves the user experience... but **it's so easy to go too far**."

Lambert's three-Llama arc:

| Model | Style outcome |
|---|---|
| Llama 2 (2023) | Over-refusals (`03`). Too much safety tuning |
| Llama 3 (2024) | "A brilliant success for its model style" |
| Llama 4 (2025) | "Too chat tuned," and an Arena controversy (`05`) |

"It's such a fine line between getting this right, but it is important to know that there can be value in doing it... **Overoptimization makes it actually hard to get there.**"

---

## Slide: "The chattiness balance: many chat evals were easy to overfit"

### The evals involved

| Eval | How it scores | Gameable by style? |
|---|---|---|
| **AlpacaEval** | An **LLM judge** compares your answer with a reference model's; you get a win rate | Very. LLM judges favor longer, more formatted answers |
| **MT-Bench** | An LLM judge rates multi-turn answers 1–10 | Very |
| **Chatbot Arena** (later LMArena) | **Real people** vote between two anonymous answers | Less, but not immune (`05`) |
| Math / code benchmarks (e.g. GSM8K, HumanEval) | Checked answers | Not by style |

The slide's point: preference tuning **reliably boosts LLM-as-a-judge chat evals**, but those gains "do not transfer proportionally to Arena or real usage."

### The trade-off, in technical reports' own words

> "However, DPO leads to improvements in human preference evaluation but degradation in benchmark evaluation." (Qwen technical report, 2023)

Lambert: "It's very easy to boost the chattiness of the model without making it better at math or instruction following." So "there was a lot of papers that just did this because they're able to say, got a state-of-the-art score. I beat GPT-4 on AlpacaEval... But even when they got tested in something like LMArena, the results didn't hold up."

The evidence on the slide:
- **Ivison et al., 2024 ("Unpacking DPO and PPO"):** preference methods **run in loops or with abundant data can trade math and coding for chat performance**. Lambert: "you can obviously boost chat performance and degrade your other things by just running preference tuning longer, by, frankly, overoptimizing it."
- **Olmo 3** shipped the checkpoint with **higher math/code/reasoning scores**, not the one that maximized LLM-judged chat benchmarks. Lambert: "we were releasing lower numbers, and that makes it so our model might not reach as far," but "it felt like the scientific correct thing to do."

**"Poison apple":** he calls the chat-score gain "low hanging fruit, which is kind of like the poison apple." Easy to take, and it makes the model worse.

*(My note: this is the `02` shape again, now with an **LLM judge** as the proxy instead of an RM. In the toy table in `02`, picture the AlpacaEval column rising forever, like the RM score, while the math column peaks and falls. Choosing the Olmo 3 checkpoint is "stop near the peak of the true metric," using math/code as the stand-in for true quality.)*

### Where the trade-off is decided (my addition)

```
preference tuning run
  checkpoint 1 ─ checkpoint 2 ─ checkpoint 3 ─ checkpoint 4 ─ ...
       │              │              │              │
       └──────────────┴── eval suite on each ───────┘
           AlpacaEval/MT-Bench:  ↑      ↑↑      ↑↑↑     ↑↑↑↑
           math / code:          ↑      ↑        –       ↓
                                        ▲
                       Olmo 3's choice: this kind of checkpoint, not the last one
```

It's a **checkpoint-selection** decision made after training, with the whole eval suite in view. No single eval number makes it for you.

---

## Slide: "Length-gamed leaderboards"

*(The figure, from the Direct Nash Optimization paper, isn't in the folder. Reconstructed from the caption: results "claiming a 7B model outperforming GPT-4 on AlpacaEval.")*

- **April 2024: Direct Nash Optimization (DNO)** reported a **7B model "beating GPT-4" on AlpacaEval.** **Self-Rewarding Language Models** reported similarly unrealistic scores. "Neither holds up in real use against frontier models."
- Lambert: DNO "is a very valid, interesting preference tuning method," but the headline result was "obviously not true if you're beating GPT-4 on a chat eval with a 7B model at the time... You were hacking some measurement."

**The main mechanism: length.** "A lot of it came through length expansion where **longer generations were judged as better by language models**." The judge (GPT-4 in AlpacaEval) prefers longer answers, so training against it (or against an RM with the same bias) makes answers longer and the win rate goes up.

**The fix:** "The gaming got bad enough that AlpacaEval **and** WildBench added **linear length corrections**" (e.g. Length-Controlled AlpacaEval).

*(My addition, how a length correction works in plain terms: collect many judge decisions and fit a simple model of "how much does the win rate go up per extra token of length difference?" Then report the win rate you'd expect **if both answers were the same length**. Example: if your model wins 60% of comparisons but its answers are much longer, and the judge typically gives longer answers a +12-point boost, the length-controlled score is about 48%. The exact AlpacaEval method is more involved, but the idea is "subtract the part of the win that length alone explains.")*

**Done right: Starling Beta.** A **k-wise reward model** (trained on rankings of several answers at a time, not just pairs) plus **PPO**, on top of **OpenChat**. It went **up 10 Arena places**, with length "that actually helps raters." Lambert: "a genuinely great little research model that didn't do this overfitting."

The cost to the field: "such a kind of **souring of the field** that you had to go through this rigmarole to know if the model was really good or if they just juiced the numbers." Lambert says his skepticism toward results on Twitter dates from this period.

**Connects to:** `../lecture-4-implementing-rl/06-loss-aggregation-strategies.md`. Lecture 4 showed one way length drifts during RL: the **loss aggregation** choice. This is a second, independent one: the **reward** (RM or LLM judge) prefers longer answers. Both show up on the same `generation/length` panel, so a rising length curve means you should check both. `05` asks why raters like length in the first place.

---

## The whole section in five lines

1. **Same base, before and after preference tuning (the library):** the same facts, restructured into definition, headers and lists. "Slightly cleaned up."
2. **Style is substance:** repackaging known information has real value (*Sapiens*, Llama 3's personality). But it's "so easy to go too far": Llama 2 over-refused, Llama 4 was over-chat-tuned.
3. **LLM-judge chat evals (AlpacaEval, MT-Bench) were easy to game.** Preference tuning run too long traded math/code for chat scores (Qwen, Ivison et al.). Olmo 3 chose the checkpoint with better math/code: "stop near the peak" applied to checkpoint selection.
4. **Length-gamed leaderboards:** DNO's "7B beats GPT-4" on AlpacaEval came largely from longer answers, which led to length-controlled evals. Starling Beta showed it could be done honestly.
5. **Length drift has two sources** to check on the same dashboard panel: the loss aggregation (lecture 4) and the reward's own length preference (this lecture).
