# YASAKANI

**English** · [日本語](README.ja.md)

![YASAKANI: 2,298 tok/s on JSONL edits with one RTX 5090. A 10.31 GB fastest model file and up to 262,144 context tokens.](docs/assets/hero-en.svg)

**Qwen3.8-27B on your PC. Fast generation, long conversations, and evidence you can inspect.**

A local AI engine built by an independent developer in Japan. Chat, edit code, or connect an OpenAI-compatible client.
Apache 2.0. Runs offline after downloading the required files.

**[Download](https://github.com/yasakani-dev/yasakani/releases/tag/v0.1.5)** · [Quick start](#quick-start) · [Benchmarks](#speed) · [Quality](#quality) · [How it works](docs/TECHNOLOGY.md) · [Support development](#support)

> **Current release: v0.1.5 (2026-11-16).** Fixes for tool calls, GPU selection, and more.
> [Changes and contributor credits](docs/RELEASE_NOTES.md) · [Report an issue](https://github.com/yasakani-dev/yasakani/issues)

| Platform | Status |
|---|---|
| Windows 11 + RTX 5090 | **Tested by the developer** |
| Other RTX 30 / 40 / 50 series, 16 GB or more | Build targets included; device testing and tuning still pending |
| Linux | Community source-build instructions; no prebuilt release |
| AMD / Mac | Future targets; not supported today |

[Driver, memory, and GPU requirements](#what-you-need)

## What it is

In the published comparison on one PC with one RTX 5090, the fastest model **outperformed eight other engines in all seven categories, with thinking off and on**.
Thinking-off results are averages; thinking-on results use historical best runs. [The tables below](#speed) separate dates, versions, and aggregation methods.

YASAKANI designs compression and computation together to reduce data movement and repeated work.
It targets Qwen3.8-27B, generating one request at a time. Choose the quality or fastest model for your workload.

## Quick start

1. Install Python 3.10 or newer and check the [requirements](#what-you-need).
2. Download **`yasakani-quickstart-0.1.5.zip`** from the [v0.1.5 release](https://github.com/yasakani-dev/yasakani/releases/tag/v0.1.5).
3. Unpack it to a simple path such as `C:\yasakani` and double-click **`START.cmd`**.

Choose a number from the menu. The launcher downloads an engine and model, runs the checks, then offers browser chat, benchmarks, and an API server.
**Start with the quality model.** For a 16 GB GPU, choose the fastest model; see the [hardware status](#what-you-need) for untested cards.

**Waiting for an answer?** The default is the longest thinking setting. In terminal chat, use `/effort medium` for shorter thinking or `/think off` to disable it.
API output limits include thinking tokens. [Full usage guide](python/README.md)

<details>
<summary>Manual installation, API server, and thinking settings</summary>

1. Download the engine zip (`yskn.dll` and the Python package) from the [release page](https://github.com/yasakani-dev/yasakani/releases/tag/v0.1.5) and one model folder from
   Hugging Face ([quality](https://huggingface.co/yasakani-dev/Qwen3.8-27B-YASAKANI-quality) or [fastest](https://huggingface.co/yasakani-dev/Qwen3.8-27B-YASAKANI-speed)). Compare the SHA-256 checksums with the ones on the release page.

```bat
hf download yasakani-dev/Qwen3.8-27B-YASAKANI-quality --local-dir D:\models\qwen38-quality
```

2. Install the Python package and check the computer, the files and the installation:

```bat
pip install .
yasakani info      --model-dir D:\models\qwen38-quality --dll D:\yasakani\yskn.dll
yasakani verify    --model-dir D:\models\qwen38-quality
yasakani selfcheck --model-dir D:\models\qwen38-quality --dll D:\yasakani\yskn.dll
```

3. Chat in the terminal, or start the server and point any OpenAI-compatible client at `http://127.0.0.1:8765/v1`:

```bat
yasakani chat  --model-dir D:\models\qwen38-quality --dll D:\yasakani\yskn.dll
yasakani serve --model-dir D:\models\qwen38-quality --dll D:\yasakani\yskn.dll
```

**Good to know before the first request**

* By default the model thinks before it answers, at its longest thinking setting. On requests that ask for a certain number of characters, or
  for a whole program in one go, the thinking alone can take thousands to tens of thousands of tokens, and no answer appears during that time.
  To shorten it: `/effort medium` or `/effort low` in `yasakani chat`, `/think off` to answer without thinking; through the API,
  `reasoning_effort` (`medium`, `low`) or `{"enable_thinking": false}` in `chat_template_kwargs`.
* The output limit (max tokens) includes the thinking. If your client's limit is small, the request ends in the middle of the thinking and the
  answer is empty. Raise the limit when the model is to think (the server's default is 32,768); for requests where even that is not enough, use
  a shorter thinking setting.
* For a long program written in one go, we suggest `medium`. In a small number of trials, `medium` most often gave code that runs as it is;
  the default (longest thinking) can think for minutes and then reach the limit, and thinking off or `low` gave more code that does not run.
* With thinking off, we use temperature 0.7, top-p 0.8 and top-k 20 in development. Temperature 0 falls into repetition more easily on long prose.

The details (the model folder, supported request fields, the loop guard, the self-check) are in [python/README.md](python/README.md).
To build `yskn.dll` yourself, see [BUILDING.md](BUILDING.md); common questions are in [docs/FAQ.md](docs/FAQ.md).

</details>

## Two models

Both are Qwen3.8-27B, converted to the YASAKANI file format and compressed. You need only one of them.

| | Quality model (recommended) | Fastest model |
|---|---|---|
| Made for | staying close to the original model, at high speed | the highest generation speed and the smallest size |
| Model file | 12.37 GB | 10.31 GB |
| Same most likely next token as the original model | 95.24% | 93.01% |
| KL divergence from the original model (smaller is closer) | 0.0122 | 0.0304 |
| Download | [Hugging Face](https://huggingface.co/yasakani-dev/Qwen3.8-27B-YASAKANI-quality) | [Hugging Face](https://huggingface.co/yasakani-dev/Qwen3.8-27B-YASAKANI-speed) |

The two quality rows are agreement with the original model's next-token prediction on 20 thinking transcripts, not task accuracy; see
[Quality](#quality). The file size is not the GPU memory needed to run the model.

The agreement and KL values above use a research FP16 cache. Released KV5 measurements are listed under [Quality](#quality).

## Speed

**Measured performance, with the conditions and records included.** In the latest thinking-off tests, the fastest model reached **2,025 tok/s** on code edits and **2,298 tok/s** on JSONL edits — about **4.6× and 3.5×** the next-fastest compared engine.

These edits copy much of the input. Their gains are larger than those on code or prose written from scratch.

Tokens written per second, on one RTX 5090 (stock 600 W, no overclocking), the same PC for every engine, one request at a time. Bigger is faster.
All speed numbers in this repository are from the RTX 5090. Other GPUs are slower, and we publish no numbers for them.

**Thinking off** (greedy; average of 2 runs per task, after leaving out runs that fell into a repetition loop or made a tool call nobody asked for;
YASAKANI measured again with 0.1.4 on November 15, 2026):

| Engine | Code | Story | Translation | Structured | Code edit | JSONL edit | Long JSONL |
|---|---|---|---|---|---|---|---|
| **YASAKANI fastest model** | **512** | **195** | **479** | 882 | **2,025** | **2,298** | **1,214** |
| **YASAKANI quality model** | 468 | 172 | 464 | **892** | 2,008 | 2,243 | 1,115 |
| MegaCapybara 1.39 Tiny | 465 | 164 | 268 | 599 | 441 | 658 | 666 |
| mkyt v0.3.2 | 395 | 181 | 374 | 555 | 423 | 634 | 639 |
| ignis v0.9.6 | 330 | 167 | 298 | 439 | 331 | 510 | 514 |
| NInfer (November 2 build) | 264 | 123 | 257 | 386 | 295 | 443 | 438 |
| vLLM 0.31.0 | 290 | 148 | 261 | 354 | 241 | 426 | 431 |
| SGLang 0.5.22 | 252 | 129 | 246 | 371 | 430 | 466 | 451 |
| llama.cpp b11700 | 242 | 133 | 232 | 337 | 244 | 360 | 362 |
| TensorFold 0.6.6 | 181 | 104 | 172 | 405 | 383 | 408 | 261 |

<details>
<summary>Thinking-on comparison (November 8 best runs, not averages)</summary>

**Thinking on** (temperature 1.0, top-p 0.95, top-k 20, min-p 0; the fastest run of each task; YASAKANI measured on November 8 with the engine
of 0.1.0 to 0.1.2 (the build of November 7), not again with 0.1.4):

| Engine | Code | Story | Translation | Structured | Code edit | JSONL edit | Long JSONL |
|---|---|---|---|---|---|---|---|
| **YASAKANI fastest model** | **337** | **287** | **497** | **536** | 667 | **1,108** | **704** |
| **YASAKANI quality model** | 299 | 242 | 428 | 480 | **743** | 1,053 | 634 |
| MegaCapybara 1.39 Tiny | 290 | 206 | 308 | 429 | 550 | 663 | 516 |
| mkyt v0.3.2 | 244 | 196 | 348 | 384 | 378 | 603 | 454 |
| ignis v0.9.6 | 211 | 130 | 276 | 277 | 278 | 485 | 327 |
| NInfer (November 2 build) | 172 | 140 | 258 | 263 | 263 | 404 | 322 |
| vLLM 0.31.0 | 131 | 112 | 197 | 195 | 187 | 369 | 253 |
| SGLang 0.5.22 | 157 | 124 | 206 | 230 | 257 | 402 | 282 |
| llama.cpp b11700 | 140 | 122 | 184 | 197 | 206 | 349 | 221 |
| TensorFold 0.6.6 | 134 | 79 | 167 | 183 | 439 | 360 | 268 |

</details>

<details>
<summary>Measurement method, trial counts, exclusions, and version differences</summary>

What these tables are, and are not:

* "Speed" is the speed of **writing the answer**: (output tokens - 1) / (time from the first to the last content chunk). The time to read the
  prompt is not included.
* The tasks are 12 published prompts (code, story, translation and structured output, 3 each) and 3 tasks of our own (code edit, JSONL edit,
  long JSONL). The output limit is 4,096 or 8,192 tokens.
* Thinking off: the average leaves out runs with a repetition loop or an unrequested tool call, by one rule applied to every engine. This can
  change how many tasks a category holds. For the fastest model's "Code" it matters: 1,089 tok/s before the exclusion, 512 after (0.1.4); we do not use
  1,089 as a figure for practical code generation. The table before the exclusion is in [docs/MEASUREMENTS.md](docs/MEASUREMENTS.md).
* **On November 14, 2026 the rule for repetition was tightened and the table was made again.** The first rule missed some answers that
  repeat whole paragraphs. The fastest model's "Story" value of 285 included such runs (the English mystery story, 454 tok/s). With the new
  rule (for story and translation: a run is left out when lines that occur more than once make up 20% or more of its lines), applied to
  every engine alike, only the "Story" column changes: fastest model 285 -> 201, mkyt 204 -> 181, vLLM 150 -> 148, llama.cpp 139 -> 133.
  The table made again, and the 12 prompts one by one, are in [tools/speed/](tools/speed/).
* The large gaps are in the 3 tasks of our own (1.8 to 4.6 times the next engine). These tasks mostly copy their input, the kind of
  output that speculative decoding predicts best; since 0.1.4 a pass that copies the input checks up to 64 tokens (below). In the four categories of the 12 published prompts the fastest model is 1.1 to 1.5
  times the next engine. For 3 of the 12 prompts (CUDA code, TypeScript code, the English mystery story) the fastest model's answer was
  a repetition loop or an unrequested tool call, so they are not compared. The quality model's "Story" value is about the same as the
  next engine's (172 against 181).
* Thinking on: for each task the fastest run is taken, then the category is averaged. YASAKANI chose from 5 runs, the other engines from 2,
  so the number of tries is not equal; the table of plain averages is in [docs/MEASUREMENTS.md](docs/MEASUREMENTS.md). Many runs ended at the
  output limit, so this is not a comparison of time-to-answer or of accuracy.
* Every engine ran its own weights, quantization, key-value cache format and speed-up settings; this is not a comparison with all settings
  made equal. For an engine measured with several settings, the fastest setting of each category is used.
* YASAKANI's two models were measured with thinking off on November 15, 2026 with the `yskn.dll` of 0.1.4 (0.1.5 has the same GPU code), and with thinking
  on on November 8 with the engine of 0.1.0 to 0.1.2 (the build of November 7). MegaCapybara with thinking off was measured on November 3; its thinking-on runs and the other engines on November 6. This is not a ranking of the latest version of every program, or of
  every use.

Everything needed to check or repeat the comparison (launch settings, prompts, the measurement scripts and the raw records) is in
[`bench/`](bench/). `yasakani bench` measures your own PC.
If you measure with a script of your own, take the token count from the server's `usage.completion_tokens` (in a stream, send `stream_options`
with `{"include_usage": true}`). A stream carries one chunk per token, but where one character is split over several tokens its text arrives
with the last of them, so the number of chunks is slightly smaller than the number of tokens.

### What 0.1.4 changed

Measured one after the other on November 15, 2026 (14:51 to 15:17) with the same clients (thinking off, 2 runs per task), against 0.1.2. tok/s:

| | Code edit | JSONL edit | Long JSONL edit (160 records) | Code, Story, Translation, Structured, Long JSONL |
|---|---|---|---|---|
| fastest model | 1,210 -> 2,025 (1.67 times) | 1,125 -> 2,298 (2.04 times) | 1,141 -> 2,099 (1.84 times) | 0.96 to 1.00 times |
| quality model | 1,128 -> 2,008 (1.78 times) | 1,146 -> 2,243 (1.96 times) | 1,096 -> 2,056 (1.88 times) | 0.97 to 1.01 times |

The 64 answers of this measurement (both models) were the same text with 0.1.3 and 0.1.4, and all but one with 0.1.2: the second run of one
prompt of the quality model with 0.1.2 changed part-way through, the rare change that 0.1.3 fixed (see [Quality](#quality)). The long JSONL edit is an extra task, not a
column of the table. In this sitting 0.1.3 also ran at 0.96 to 1.01 times the speed of 0.1.2 on the other tasks, and 0.1.4 at 0.99 to 1.00 times the speed of 0.1.3; in a measurement in the morning of the same day (with the first build of 0.1.4) the three versions were within 1% of each other on these tasks. We do not know the cause of the difference. Records: [bench/20261115/](bench/20261115/); tables:
[tools/speed/SPEED_TABLES_0114.md](tools/speed/SPEED_TABLES_0114.md); how the copying works: [docs/TECHNOLOGY.md](docs/TECHNOLOGY.md) (5.4).

</details>

### One edit, start to finish

A screen recording without cuts shows the fastest model rewriting a Python file of about 20,000 characters: 5.56 s from sending the request to
the last output (4,712 tokens, 1,102 tok/s while writing, measured on November 8 with the engine of 0.1.0 while the screen was being recorded). The generated file gave the same
results as the original on 17,701 inputs. The recording, logs, the generated code and the checks are in the release's evidence archive;
see [docs/MEASUREMENTS.md](docs/MEASUREMENTS.md).

[Watch the uncut demo (November 8, v0.1.0)](https://github.com/yasakani-dev/yasakani/releases/download/v0.1.5/fast_live_uncut.mp4)

## Quality

The models are compressed (the original BF16 weights are 53.8 GB), so they are **not identical** to the original Qwen3.8-27B.

How close the next-token predictions stay to the original BF16 model was measured on 20 thinking transcripts from our own use of the model,
none of them used when the files were made (20,460 positions, FP16 key-value cache, November 8, 2026). Thirteen configurations of different
engines and compressed files were measured on the same texts:

* The **quality model** has the smallest KL divergence (0.0122) and the highest top-1 agreement (95.24%) of the thirteen.
* Against the GSQ-RCO files of similar size run by llama.cpp: 12 GB class, 95.24% against 93.64% (IQ3_S); 10 GB class, the fastest model's
  93.01% against 91.22% (IQ3_XXS). The KL divergence is about 47% and 41% smaller.

Agreement is "chose the same most likely next token as the original model". It is **not** the rate of correct answers on tasks, and the
result covers thinking text only: on general web text an additional evaluation showed a KL divergence 3-10% larger for the quality model
than for our earlier conversion of the same size. The full table, including the configurations for which only a lower bound of the KL
divergence could be measured, is in [docs/MEASUREMENTS.md](docs/MEASUREMENTS.md).

<details>
<summary>Released KV5 results, reproduction, and output-consistency checks</summary>

**Checking it yourself.** The numbers above were measured with a research build of the engine and an FP16 key-value cache. The released
`yskn.dll` has no FP16 cache: it always rounds the cache with KV5. Measured again with the engine of releases 0.1.0 to 0.1.2 as the
package starts it (KV5 cache, November 14, 2026), the quality model gives a KL divergence of 0.0126 and a top-1 agreement of 95.35%, the fastest model 0.0309 and 92.98%. The
20 texts, a program that makes the reference from the original model with Hugging Face transformers, and the program that measures a
released model are in [tools/quality/](tools/quality/). The file format is specified in [docs/FORMAT.md](docs/FORMAT.md), and
[tools/yskn_decode.py](tools/yskn_decode.py) reads a model file without the engine and compares the stored weights with the original
ones. The program that made the model files from the original weights is in [tools/convert/](tools/convert/). Run again on the PC
that made them, with only the output matrix taken from the released files, it gave the released files byte for byte
([its README](tools/convert/README.md)).

Speculative decoding only keeps tokens that the main model agrees with, so it is built not to change what is written. Up to 0.1.2, while
other programs were using the GPU at the same time (the desktop was drawing), a repeated request now and then had the same tokens as the
others up to one position and continued differently from there (November 10, 2026: 1 of 190 runs with the default settings and 21 of 432
with speculative decoding switched off; November 11 and 12: 10 of 593 runs). With the GPU otherwise idle, all 240 repeated runs were identical.
On November 11 we found the cause: in the kernels that apply the position rotation to queries and keys, one synchronization was missing,
so a value could be overwritten before another thread had read it. 0.1.3 adds it. In our tests with builds that have this fix, the change
did not appear again (for example 0 of 2,800 passes compared bit for bit while another process kept the GPU busy, against 6 of 3,200
without the fix). We still do not promise bit-identical text in every condition. The details are in [docs/MEASUREMENTS.md](docs/MEASUREMENTS.md).

The wider copy passes of 0.1.4 were checked against plain decoding. A pass of 17 to 64 rows gave bit for bit the same choices and values as
passes of 16 rows (30 of 30 checks), and 255 of 255 generations with copying (fastest model greedy and with a fixed seed, quality model
greedy; the settings of 0.1.4, the same with parts switched off, and the 0.1.3 engine) gave the same tokens as decoding one token per pass (28 of them used wide passes). A prompt continued from a
saved state gave the same output as the same prompt read from the start (37 of 37 cases).
0.1.4 also makes the point where `thinking_budget` or the loop guard cuts the thinking depend only on the generated tokens. Up to 0.1.3 it
could move by the tokens confirmed before the cut was noticed, so the answer could depend on the look-ahead.

</details>

## Long contexts

The context length goes up to 262,144 tokens. The key-value cache is stored in YASAKANI's own compressed format, **KV5**. With the fastest
model and the full 262,144-token context reserved, the engine takes 16.78 GiB with KV5 against 27.13 GiB with an FP16 cache (about 38% less
for the whole engine, about 66% less per cached token). With the fastest model on fixed texts of 2K to 128K tokens,
the KL divergence with KV5 was at most 0.00052 higher on average than with FP16 (the FP16 side was a research build). Against a plain
8-bit cache (a measurement build of the 0.1.2 source, context 2,048 tokens), the KL with KV5 was about 3% higher for the quality model and
about 2% higher for the fastest model ([docs/MEASUREMENTS.md](docs/MEASUREMENTS.md)).
The 262,144 figure is what the engine reserves at start; quality was checked up to 128K. On a GPU with less
memory than the RTX 5090 the context is made to fit (see [What you need](#what-you-need)).

Together with the helper model that speeds up generation, the GPU memory in use at the full 262,144-token context was about 19.1 GiB for the
fastest model and about 21.2 GiB for the quality model (about 14.3 GiB and 16.4 GiB with `--max-ctx 32768`; November 11, 2026, the difference
of the nvidia-smi values before and after loading).

<details>
<summary>Reusing a long document: first-token latency and conditions</summary>

**Reusing a long prompt.** When a request repeats the previous prompt, continues the conversation or sends back a tool result, the
standard server reuses what it has read and reads only the new part: after a 229,376-token document the first token
came 0.35 to 0.38 s after the request, against 116 s the first time (0.1.1 to 0.1.4, measured on November 15, 2026). New in 0.1.4: the same
document followed by a different question is not read again either (0.37 s; 0.1.1 to 0.1.3 read it again, 116 s), when the
document comes before the question and the question part is up to about 4,000 tokens. The engine keeps up to 8 saved states along the
prompt of one conversation in pinned host memory (about 1.5 GiB of RAM; no more GPU memory; current defaults, not tuned). It reads from the start when an unrelated request came
in between, when thinking is switched on or off, after a request with an image, and when a prompt shorter than 8,192 tokens changes part-way
([docs/TECHNOLOGY.md](docs/TECHNOLOGY.md), chapter 8; [docs/MEASUREMENTS.md](docs/MEASUREMENTS.md)). The harness runs its own copy of the
Python package, so it does not reuse a document for a different question yet.

</details>

## How it works, briefly

| Less of… | How | Where it helps |
|---|---|---|
| Data moved from GPU memory | Design the compressed weight format and its computation together | Every generated response |
| Repeated computation | Verify proposed tokens together; verify up to 64 copied tokens in a pass | Code and JSONL edits |
| Prompt rereading | Reuse the shared part of the previous request | Follow-up questions on long documents |
| Memory for conversation history | Compress the key-value cache with KV5 | Long conversations and large contexts |

The runtime combines these ideas with kernels tuned on the RTX 5090. Performance on other GPUs will be published after testing on those devices.
See the [technical notes and source map](docs/TECHNOLOGY.md), [file-format specification](docs/FORMAT.md), and [published conversion tools](tools/convert/).

## What you need

| | |
|---|---|
| GPU | NVIDIA **GeForce RTX 5090** (32 GB): tested. Since 0.1.3 the engine is also built to run on other NVIDIA RTX 30, 40 and 50 series cards with compute capability 8.6 or higher and 16 GB or more (below), but it is not optimized or tested on them |
| OS | **Windows 11**, 64-bit (the only version tested) |
| Driver | NVIDIA driver 580 or newer (a driver for CUDA 13; tested with 616.92). The quickstart checks it before downloading anything and stops if it is older |
| Software | Microsoft Visual C++ 2015-2022 x64 runtime, Python 3.10 or newer |
| Disk / RAM | 20 GB free (one model is a download of about 13 to 15 GB); 32 GB of RAM recommended (tested on a PC with 96 GB) |

**Other GPUs (since 0.1.3).** The engine is built for compute capability 8.6, 8.9 and 12.x, and is meant to run on these cards (none of them has been tried yet):

| Memory | Cards | Models |
|---|---|---|
| 24 GB | RTX 3090, RTX 3090 Ti, RTX 4090 | both models |
| 16 GB | RTX 4060 Ti 16GB, RTX 4070 Ti SUPER, RTX 4080, RTX 4080 SUPER, RTX 5060 Ti 16GB, RTX 5070 Ti, RTX 5080 | the fastest model only, with a shorter context |

We have not optimized the engine for these cards and have not tested it on them (we have only an RTX 5090), so neither its speed nor
its operation is guaranteed there. They are slower than the RTX 5090. Since 0.1.4 `yasakani selfcheck` (the quickstart runs it) compares
the card's predictions with those of the RTX 5090 and fails when they differ by more than small differences in the computation; every GPU
also checks that its passes of different widths agree. On a card with less than 32 GB the context length is chosen from
the free GPU memory each time the model starts (at most 262,144 tokens; on a 16 GB card about 65,000 to 82,000 tokens when the display
and other programs use 1 GB of it - calculated from measurements on the RTX 5090, not measured on these cards; the draft model's output
table `bdraft2_k8.shead` is then left out, which made each pass about 3% slower on the RTX 5090). Testing and tuning on
each of these cards is a goal of the crowdfunding campaign; see [Support](#support). Results from your card are welcome
([CONTRIBUTING.md](CONTRIBUTING.md)).

Not supported: AMD GPUs, Macs, NVIDIA cards with less than 16 GB, cards older than the RTX 30 series (compute capability below 8.6),
other models.

**Linux:** the engine can be built from source (contributed in [#3](https://github.com/yasakani-dev/yasakani/pull/3)); the steps and the tested setups are in
[BUILDING.md](BUILDING.md#linux-building-from-source). This is building from source, tested by contributors, not an official release: there is no
prebuilt Linux release or launcher yet. An official Linux release (prebuilt files, a launcher, checks on plain Linux, and support) is one of the
crowdfunding goals.

<details>
<summary>Optional image input: setup and limitations</summary>

**Images (optional):** since 0.1.2 the server can also take images on Windows (a community contribution, [#4](https://github.com/yasakani-dev/yasakani/pull/4)). This needs
files that are not part of the release: the image encoder `mmproj-Qwen3.8-27B-BF16.gguf` (0.93 GB), the CPU build of llama.cpp b10488,
a Qwen3.8-27B GGUF file (only its settings and vocabulary are read, but the file is large: about 17 to 19 GB for Q4_K_M) and Pillow.
Each image is encoded on the CPU, after it is downsized to at most 1,024 image tokens (about one million pixels). On our PC (AMD Ryzen 9 9950X3D2, 16 cores,
16 threads) that took about 15 s for a large image and 1.2 s for a 512 × 512 one; while an image is encoded, the server answers
no other request. We checked that it works on our RTX 5090 with the quality model and a few
simple test images; we have not measured how well the model understands images. Set-up and limits: [python/README.md](python/README.md#image-input-optional).
The quickstart menu and the harness do not use images. On Linux image input works with llama.cpp b10488 built from source
([BUILDING.md](BUILDING.md#image-input-on-linux); contributed in [#8](https://github.com/yasakani-dev/yasakani/pull/8), tried by its author on native Ubuntu 26.04, not by us).

Close other programs that use a lot of GPU memory (games, other LLM servers) before you start. When the memory runs out there is no error:
generation just becomes several times slower (a warning is printed at start in that case).

</details>

## Use it as an agent

**An optional browser agent interface is available.** Its bundled server is older and does not include the latest tool-call fixes or automatic context sizing.

<details>
<summary>Installation, bundled version, and the 24 GB GPU requirement</summary>

`yasakani-harness-0.1.0.zip` on the [release page](https://github.com/yasakani-dev/yasakani/releases/tag/v0.1.5) opens an agent screen in your browser, run by the model
on your PC: the model reads and writes files in a work folder and runs commands (you choose on the screen whether it may change files).
Run the quickstart above first; then unpack the harness zip into the quickstart folder or next to it and double-click `START_HARNESS.cmd`.

* The screen is [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness) (open-source software by DeepSeek, MIT license), used
  unchanged. It is installed from npm on first use (about 730 MB; Node.js is fetched too if the PC has none). YASAKANI adds a settings file
  and a Japanese language pack. YASAKANI is not affiliated with DeepSeek, and this combination is not an official product.
* The harness zip is the one of 0.1.0, unchanged. It starts the server with the copy of the Python package that comes with it (0.2.0.dev0);
  the tool-calling code of that copy is that of the standard package of 0.1.1; the tool-call fixes of 0.1.5 ([#12](https://github.com/yasakani-dev/yasakani/pull/12), [#21](https://github.com/yasakani-dev/yasakani/pull/21)) are not in it. It does not have the image input of 0.1.2.
  It does not choose the context from the GPU memory either (0.1.3): it opens the full 262,144 tokens, so it needs a card with 24 GB or more.
* A new session starts with the model's longest thinking setting. One reply may take up to 98,304 tokens; a thinking longer than 49,152
  tokens is closed there and the model goes on to the answer. With both models installed, the screen can switch between them.
* Everything is kept inside the harness folder. A DeepSeek Harness you already use is not touched.

What it does, what was checked and what was not: [harness/HARNESS_README.txt](harness/HARNESS_README.txt).

</details>

## Documentation and contributions

| I want to… | Start here |
|---|---|
| Use chat, the API, or tools | [Python package](python/README.md) |
| Reproduce the results | [Benchmark records](bench/README.md) · [Methodology](docs/MEASUREMENTS.md) · [Quality evaluation](tools/quality/) |
| Build the engine | [Windows / Linux build guide](BUILDING.md) |
| Understand the design | [Technical notes](docs/TECHNOLOGY.md) · [File format](docs/FORMAT.md) |
| Fix a problem or contribute | [FAQ](docs/FAQ.md) · [Contributing](CONTRIBUTING.md) · [Security](SECURITY.md) |

Results from your GPU are welcome. Include the GPU, driver, model, context length, and whether your timing includes prompt processing.

## Limits

* Windows 11 + Qwen3.8-27B only (Linux: build from source). Tested on the RTX 5090 only; the build also targets other RTX 30, 40 and 50 series cards with 16 GB or more, untuned and untested. One request at a time. Images only through the optional image input of the server (new in 0.1.2; on Linux only with llama.cpp built from source, not tried by us; [python/README.md](python/README.md#image-input-optional)); no `n` above 1, no `logprobs`. Tool / function calling: `tools` with `tool_choice` `auto` or `none` (new in 0.1.1); `required` or a named function is accepted since 0.1.4 but not enforced, so an answer can come without a call. It is new and has been tested far less than plain chat.
* The release files are not code-signed, so Windows may warn about them (see [SECURITY.md](SECURITY.md)).
* The answers are not identical to the original model's (see Quality).

## Support

Development is funded by a crowdfunding campaign on CAMPFIRE: https://camp-fire.jp/projects/992233/view (until December 16, 2026). This
release does not depend on its result. The support goes to test hardware, electricity and development time: testing and tuning on the other
RTX cards first (24 GB and 16 GB; 0.1.3 is built for them, but untested and untuned), then AMD and Mac, in the order and under the conditions
written on the campaign page.
Supporters who wish to be named are credited in this repository, and in the startup credits from a later version. News and
measurements: https://x.com/yasakani_dev

**Outside Japan:** the CAMPFIRE campaign is only for backers living in Japan (CAMPFIRE's rules do not allow digital rewards, like this
campaign's, for backers abroad). From abroad you can support the development through GitHub Sponsors:
https://github.com/sponsors/yasakani-dev - there are no rewards, and the support goes to the same purpose, test hardware for more GPUs.
Starring this repository, trying the engine and reporting what you measure helps a lot too.

## License

Apache License 2.0 - see [LICENSE](LICENSE), [NOTICE](NOTICE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). The model files are
derived from Qwen3.8-27B (Apache-2.0, Alibaba Cloud) and from a publicly released draft model (Apache-2.0, z-lab); they are not the
original weights.
