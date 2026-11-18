# FAQ

日本語は [FAQ.ja.md](FAQ.ja.md)。

## What is YASAKANI?

An inference engine for running **Qwen3.8-27B** on your own Windows PC with **one NVIDIA RTX GPU** (tested on the RTX 5090). It is GPU kernels, a file format for the
compressed model, and a runtime, with an OpenAI-compatible server, a terminal chat, a benchmark and a self-check. One person in Japan built it. It is
not a general-purpose engine: one model, tuned on one card.

## Which model should I download?

There are two conversions of Qwen3.8-27B, and you need only one. The **quality model** (12.37 GB) stays closest to the original model and is the
one recommended for everyday use. The **fastest model** (10.31 GB) is smaller and writes faster in most kinds of task, and is a little further
from the original model. The numbers for both are in the [README](../README.md).

## How fast is it, and how was that measured?

On one RTX 5090 with Qwen3.8-27B, YASAKANI's fastest model wrote answers faster than each of the eight other engines we measured on the same PC,
in all seven task categories, with thinking off and with thinking on. For example, with thinking off it wrote 2,025 tokens per
second on the code-edit task (0.1.4, November 15), against 441 for MegaCapybara 1.39 measured on November 3 (about 4.6 times), and
1,214 on long JSONL.

"Speed" is the speed of **writing the answer**: (output tokens - 1) / (time from the first to the last content chunk). The time to read the prompt
is not included. Each engine ran with its own weights, quantization, key-value cache format and speed-up settings, one request at a time. With
thinking off, runs that fell into a repetition loop or made an unrequested tool call are left out of the average by one rule for all engines;
with thinking on, the fastest run of each task is used, chosen from 5 runs for YASAKANI and from 2 for the other engines. The tables are in the
[README](../README.md); the conditions, the tallies without these choices and what the tables do not show are in
[MEASUREMENTS.md](MEASUREMENTS.md). [`bench/`](../bench/) has what you need to check or repeat the comparison, and `yasakani bench` measures
your own PC.

## Will I see the same speed?

Not exactly. The speed depends on what you ask: predictable text (code, structured output) is written faster than free-form prose, because the
engine proposes several words at once and keeps the ones the main model agrees with. Your GPU's clocks, cooling and the other programs using it
matter too. Run `yasakani bench` to see your own numbers.

## How do I measure the edit speed on my PC?

```bat
yasakani bench --edit --model-dir D:\models\qwen38-quality --dll D:\yasakani\yskn.dll
```

After the quickstart, open a command prompt in its folder, run `yasakani_env.cmd` (it sets the model folder and the DLL), then type
`yasakani bench --edit`. It runs the three edit tasks of the speed tables: the code edit and the JSONL edit of the README table, and the
long JSONL edit (160 records) of "What 0.1.4 changed", with the prompts, output limits and expected answers of [`bench/`](../bench/) (they
come with the package). As in the tables it is greedy with thinking off. `--think` turns thinking on, with temperature 1.0 and an output
limit of 32,768 tokens for every task (the server's default), so tok/s varies from run to run and the thinking part is not sped up much;
the thinking columns of the README tables are not what it reproduces. `--runs 2` repeats each task, `--json` prints a machine-readable
result. For each task it prints the tokens, the seconds, tok/s, the tokens per pass of the main
model, the number of wide copy passes, and whether the answer is the expected one. The first lines give the package version, the SHA-256 of
`yskn.dll`, the GPU (name, compute capability, driver), the model file and whether wide copy passes are used, so a pasted result says what
it was measured with.

The wide copy passes (since 0.1.4: a pass that copies the input checks up to 64 tokens) are used on compute capability 12.x (RTX 50 series)
only, and were measured on the RTX 5090 only; other GPUs copy with passes of up to 16 tokens. Speed here is (tokens - 1) / the engine's
time from the first to the last generated token, inside the process; the thinking-off tables were measured through the server's stream, so
the numbers are close to them, not the same. The file of the code edit is CPython's `textwrap.py` with one e-mail address replaced by
`<email>`, as in `bench/`. YASAKANI's thinking-off measurements of November 15 (the speed table and "What 0.1.4 changed") used this same
text; every earlier record of the code edit in `bench/` (November 3 to 8; among them the other engines' rows and YASAKANI's thinking-on
rows) had the address itself, 4 prompt tokens more. In the records of November 15 (0.1.4) every answer was the expected one except the
quality model's code edit (2 lines differ).

## Are the answers the same as the original model's?

No. The models are compressed (the original BF16 weights are 53.8 GB; the quality model is 12.37 GB and the fastest model 10.31 GB), so they
are **not identical** to the original. On 20 thinking transcripts that were not used when the files were made (20,460 positions, FP16 key-value
cache, measured November 8, 2026), the most likely next token is the same as the original BF16 model's **95.24%** of the time for the quality
model and **93.01%** for the fastest model, and the KL divergence is 0.0122 and 0.0304. These numbers say how close the next-token predictions
are; they are not the rate of correct answers on tasks, and they cover thinking text. On general web text an additional evaluation showed a KL
divergence 3-10% larger for the quality model than for our earlier conversion of the same size.

Does the speed-up itself change the answers? It is built not to: speculative decoding only keeps tokens that the main model agrees with.
Up to 0.1.2, while other programs were using the GPU at the same time, a few repeated runs had the same tokens as the others up to one
position and continued differently from there (November 10, 2026: 1 of 190 runs with the default settings, 21 of 432 with speculative
decoding switched off; 256 tokens each). With nothing else using the GPU, all 240 repeated runs were identical. The cause was a missing
synchronization in the kernels that apply the position rotation; 0.1.3 fixes it, and in our tests with the fix it did not appear again.
We still do not promise bit-identical text in every condition; the details are in [MEASUREMENTS.md](MEASUREMENTS.md).

The copying of 0.1.4 was checked against decoding one token per pass: 255 of 255 generations with copying (28 of them with wide passes of
up to 64 rows) gave the same tokens ([TECHNOLOGY.md](TECHNOLOGY.md), chapter 7).

## Which GPUs, which systems?

We test on **Windows 11 and the RTX 5090** (32 GB). Since 0.1.3 the engine is also built to run on other NVIDIA RTX 30, 40 and 50 series cards with
compute capability 8.6 or higher and 16 GB or more: the RTX 3090, RTX 3090 Ti and RTX 4090 (24 GB; both models) and the
RTX 4060 Ti 16GB, RTX 4070 Ti SUPER, RTX 4080, RTX 4080 SUPER, RTX 5060 Ti 16GB, RTX 5070 Ti, RTX 5080 (16 GB; the fastest model only, with a shorter context). The engine is
not optimized for them and we have not tested it on them, so neither its speed nor its operation is guaranteed; they are slower than the
RTX 5090. The quickstart says so and goes on. Since 0.1.4, however, its self-check compares the card's predictions with those of the
RTX 5090, and the quickstart stops when they differ by more than small differences in the computation, or when the card's passes of
different widths do not agree.
Results from these cards are welcome (see [CONTRIBUTING.md](../CONTRIBUTING.md)).
AMD GPUs, Macs, cards with less than 16 GB and cards older than the RTX 30 series are not supported. On Linux the engine can be built from source (a community contribution, [#3](https://github.com/yasakani-dev/yasakani/pull/3), checked on
WSL2 / Ubuntu 24.04 with an RTX 5090); a prebuilt Linux release with a launcher is one of the campaign's goals. Testing and tuning on the 24 GB and 16 GB RTX cards first, then AMD and Mac - that is what the
crowdfunding campaign is for; the order and the conditions are on the [campaign page](https://camp-fire.jp/projects/992233/view).

The NVIDIA driver must be 580 or newer. The engine is built with CUDA 13.2; 580 is the oldest driver NVIDIA lists for programs of the CUDA
13 toolkits under its minor version compatibility
([NVIDIA's documentation](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)), with limits: features that
need a newer driver fail, and so does compiling PTX at load time. The engine carries machine code for each GPU generation it supports, so
it needs no PTX compilation. We have tested 616.92 only.
The quickstart checks the driver before it downloads the engine or a model, and if the driver is older it stops and shows where to get a
new one.

Two GPUs in one PC: one process uses one GPU, chosen with `--device N` (GPU N of `yasakani info` and nvidia-smi). With 0.1.5 this
could fail with `graph instantiate failed` while the other GPU was visible ([#20](https://github.com/yasakani-dev/yasakani/issues/20)).
0.1.6 has a fix for this. On November 18, 2026 the reporter of #20 confirmed that it works on a PC with two GPUs (RTX A2000 and
RTX 5060 Ti); we have not tried two GPUs ourselves. A `CUDA_VISIBLE_DEVICES` you set yourself is kept. See
[What you need](../README.md#what-you-need).

## How much memory and disk does it need?

One model folder is a download of about 13 GB (fastest model) or 15 GB (quality model); keep 20 GB free. The context length goes up to 262,144
tokens. With the fastest model and that full context reserved, the engine takes 16.78 GiB of VRAM (27.13 GiB if the key-value cache were kept
in FP16); this is the amount reserved at start, not the peak of the whole setup. 32 GB of system RAM is recommended.

Together with the helper model that speeds up generation, the GPU memory in use at the full 262,144-token context was about 19.1 GiB for the
fastest model and about 21.2 GiB for the quality model (about 14.3 GiB and 16.4 GiB with `--max-ctx 32768`; November 11, 2026).

On a GPU with less than 32 GB, the context is chosen from the free GPU memory each time the model starts (at most 262,144 tokens;
`--max-ctx` sets it yourself). On 16 GB cards only the fastest model fits. The rule is in
[python/README.md](../python/README.md#gpus-other-than-the-rtx-5090).

## It is much slower than the tables

The tables were measured on an RTX 5090; other GPUs are slower, and we have not measured them. Check the GPU memory first. When other programs (games, other LLM servers) use GPU memory and it runs out, there is no error: generation just
becomes several times slower. In that case a warning that the GPU memory is almost full is printed at start. Close the other programs, or
lower `--max-ctx`. The speed also depends on what you ask (see "Will I see the same speed?" above).

A monitor on the same GPU can also cost a lot. In one report ([#18](https://github.com/yasakani-dev/yasakani/issues/18)), an RTX 5070 Ti
that also drove two monitors (one of them at 240 Hz) generated 1.5 to 2.1 times as fast once the monitors were moved to the CPU's
integrated graphics: about half the speed of the RTX 5090, as its memory bandwidth suggests. The tokens per pass were the same; with
the monitors on the card each pass took longer. If your PC has integrated graphics, connecting the monitors to the motherboard leaves
the GPU to YASAKANI (`nvidia-smi --query-gpu=display_active --format=csv` then says `Disabled`). The cause is not known yet, and we
have not measured it on other cards.

## It only thinks and never answers

By default the model thinks before it answers, at its longest thinking setting. On requests that ask for a certain number of characters, or for
a whole program in one go, the thinking alone can take thousands to tens of thousands of tokens. The output limit (max tokens) includes the
thinking, so with a small limit the request ends in the middle of the thinking and the answer is empty. Raise the limit (the server's default is
32,768) or shorten the thinking: `/effort medium`, `/effort low` or `/think off` in `yasakani chat`; `reasoning_effort` (`medium`, `low`) or
`{"enable_thinking": false}` in `chat_template_kwargs` through the API.

## Is a long prompt reused?

When a request repeats the previous prompt, continues the conversation or sends back a tool result, the standard server reuses what
it has read and reads only the new part: after a 229,376-token document the first token came after 0.35 to 0.38 s, against 116 s the first
time (measured with 0.1.1 to 0.1.4). Since 0.1.4 the same document followed by a different question is reused too (0.37 s), when the
document comes before the question and no other request comes in between. The prompt
is read from the start after an unrelated request, when thinking is switched on or off, after an image, and when a prompt under 8,192
tokens changes part-way. `usage.prompt_tokens_details.cached_tokens` shows how many tokens were reused; the details are in
[MEASUREMENTS.md](MEASUREMENTS.md).

## How do I connect another app?

Any app that can talk to an OpenAI-compatible server: set the address to `http://127.0.0.1:8765/v1` and the model name to `qwen3.8-27b` (any
API key will do). If it does not connect:

* An app inside Docker, or anything outside this computer: by default only connections addressed to `127.0.0.1` are accepted. Start the server
  with `yasakani serve --host 0.0.0.0 --api-key YOUR_KEY` and give the app that key.
* Apps that call the server straight from a web page: calls from other pages are refused, for safety.
* Apps that depend on tool / function calling (coding agents, for example): supported since 0.1.1. The server takes `tools` with
  `tool_choice` `auto` or `none` and returns the calls as `tool_calls`. Since 0.1.4 it also accepts `required` or a named function, but does
  not enforce the call: the model decides, and an answer without a call comes back with `finish_reason` `stop`. The old `functions` field is
  answered with 400. Version 0.1.0 answered every request with `tools` with 400: if you use 0.1.0, put a newer quickstart (0.1.1 or later) into the same folder
  and run `START.cmd` again (the model is not downloaded again). Tool calling is new and has been tested far less than plain chat.
  An agent screen is in the harness (see "Development and support" in the [README](../README.md)).

## Several requests at once (experimental)

All numbers in this section were measured on one RTX 5090 (32 GB) with a development build of 0.2.0, November 17 and 18, 2026. Full conditions: [MEASUREMENTS.md](MEASUREMENTS.md#several-requests-at-once-experimental).

### How do I turn it on?

Start the server with `--parallel N` (or set `YASAKANI_PARALLEL=N`):

```bat
yasakani serve --model-dir D:\models\qwen38-quality --parallel 4
```

The default is 1: one request at a time, the same path as before. The first time the server starts with an engine library, a GPU and a driver, it runs N short requests at once and compares their tokens with the same requests sent one by one. If they differ, it serves one request at a time and says so. The result is saved, so later starts skip the check. `yasakani selfcheck --parallel N` runs the same check on demand. `yasakani bench --concurrency N` measures N requests at once on your GPU.

### When does it help, and when not?

It helps when several different requests write new text. It does not help with edits, where most of the answer is a copy of the input. One request alone checks up to 64 copied tokens per pass on RTX 50 cards, but a shared pass carries at most 16 rows for all requests together. We think this is why.

Fastest model, thinking off, temperature 1.0, top-p 0.95, top-k 20, seeds 1 to 16, `--parallel 16 --max-ctx 16384`. Each task was sent 16 times one by one, then N of those at once. The number is (all output tokens / time from the first send to the last end) at once, divided by the same for one by one. Mean of 2 runs, one uncounted request first.

| Task | N = 4 | N = 16 |
|---|---|---|
| Story: Chinese dialogue | 2.69 | 6.56 |
| Story: Chinese science fiction | 2.73 | 6.12 |
| Story: English mystery | 2.48 | 4.11 |
| Translation: Markdown | 2.18 | 3.02 |
| Code: Python | 2.14 | 2.98 |
| Code: CUDA | 1.96 | 2.63 |
| Translation: English to Chinese | 2.00 | 2.54 |
| Code: TypeScript | 1.67 | 2.53 |
| Structured: SQL | 1.94 | 2.19 |
| Translation: Chinese to English | 1.88 | 2.17 |
| Structured: CSV | 1.85 | 2.03 |
| Structured: JSONL | 1.17 | 1.11 |
| Long JSONL (own task) | 0.95 | 0.92 |
| Code edit (own task) | 0.54 | 0.53 |
| JSONL edit (own task) | 0.55 | 0.50 |
| Mixed: 4 code edits, 4 JSONL edits, 8 published prompts | | 0.99 (16 at once) |

Below 1 means slower than one by one. With N = 2 the edits were 0.42 and 0.59. Each single request is always slower than when it runs alone. The quality model was not measured this way.

### Are the answers the same as one at a time?

Yes, under two conditions. The request has its own `seed`, and its prompt is read the same way, which `timings.snap_mode` shows (0 from the start, 1 continuing the request before it, 2 from a saved state). With a stop string the answer is the same up to the stop. We compared the token ids, the finish reason, tool calls, text, thinking and the token counts.

### How much GPU memory does it need, and how long can the context be?

All requests get the same context length. Without `--max-ctx` the server picks it from the free GPU memory and prints it at start. It reserves N + 1 contexts of 22,400 bytes per token and about 256 MiB per request. For example, one 65,536-token context with its request is about 1,656 MiB. If not even 16,384 tokens fit, the server stops and names the largest N that fits.

Measured on the 32 GB RTX 5090, with the desktop using about 1.8 to 1.9 GB:

| N | Quality model | Fastest model |
|---|---|---|
| 1 | 262,144 | 262,144 |
| 2 | 196,608 | 229,376 |
| 4 | 114,688 | 131,072 |
| 8 | 65,536 | not measured |
| 16 | 16,384 | not measured |

With the quality model all four ran under load without running out of memory. The memory used stayed 297 to 524 MiB below the server's estimate.

### What do `--keep-turn` and `--max-wait` do?

When an answer ends with tool calls, the client usually sends the next turn with the tool results soon after. `--keep-turn` keeps that conversation's place for a while. The next turn then continues where the answer stopped and does not read the whole conversation again. While its place is kept, other requests cannot use it.

`--max-wait` limits how long a new request waits because of kept places and next turns. After that time it takes the next kept place, and that conversation reads its prompt again. The time it spends waiting behind requests that came before it does not count.

Both default to `auto`. `auto` keeps a place about as long as reading the conversation again would take, from 1 to 30 seconds (1 second for a short conversation, about 4 seconds for 16,000 tokens). The wait limit is that time plus 0.25 seconds. So a kept place never holds others back much longer than the time it saves. Use `--keep-turn off` to keep nothing, or give seconds.

Comparison on one load: fastest model, `--parallel 4 --max-ctx 65536`, thinking off. 6 agents read a file in 8 parts through a tool (the tool took 0.05, 0.5, 3 or 15 seconds, 2 times each), so their conversations grew to 8,719 to 40,957 tokens. At the same time 3 chat users sent short streamed requests every 1 to 6 seconds. Mean of 3 runs.

| Setting | All agents done | Conversations read again | Chat, first token, p95 | Total tok/s |
|---|---|---|---|---|
| keep nothing (`off`) | 178.1 s | 41 times, 154.0 s | 21.8 s | 22.6 |
| keep 2 seconds, no wait limit | 123.3 s | 24 times, 84.6 s | 25.8 s | 31.7 |
| `auto` (default) | 94.2 s | 19 times, 45.9 s | 17.2 s | 42.3 |
| `auto`, after the last fix of the wait limit | 100.6 s | 23 times, 56.5 s | 15.6 s | 40.6 |

The last row was measured later the same day, after the wait limit was fixed to count all waiting caused by next turns. The agents got 5% slower and the chats faster. The chat p95 stays high in all settings because reading a long prompt pauses everything. In a second load (`--parallel 2 --max-ctx 32768`, agents that call a tool 30 times in a row and send each next turn 20 ms after the answer, plus new requests) the wait limit cut the longest wait of a new request from 7,977 ms to 1,529 ms (fastest model) and from 7,448 ms to 1,322 ms (quality model).

### What happens when every place is busy?

New requests wait in a queue. When more than `--max-queue` requests wait (default 64), the server answers 503 with `Retry-After: 2`. `/health` shows `parallel`, `slots_busy`, `slots_idle` and `queued`. `timings` shows `queue_ms` (the time the request waited).

### Can I send images?

Yes. A request with images runs alone: the other requests pause until it is done. In our tests, all requests with images gave the same answers as one at a time.

### What happens when a client disconnects?

The request stops and frees its place. This works while it waits, while it streams, and while it generates without a stream. A request with a second part (after a thinking budget or a loop cut) keeps its place for that part for 2 seconds after the client got the first part, and up to 30 seconds for a slow client.

### Do seeds work?

Yes. A request with a `seed` gets the same tokens as alone. A request without a `seed` takes the next seed in the order requests arrive, as in one-at-a-time serving. Its result therefore depends on what else was sent.

### What was tested, and what was not?

Tested on the RTX 5090 only, with both models:

* One at a time (`--parallel 1`), with a development build that has the same GPU code as 0.2.0: the same tokens as 0.1.5 on 35 prompts per model, and within 1% of its speed on 15 tasks (thinking off).
* With the 0.2.0 build: the same tokens as 0.1.6 on 35 prompts per model, and a speed difference of -0.02% (fastest) and -0.07% (quality) on the same 15 tasks.
* Several at once: about 35,000 requests compared with the same conversation sent alone. None differed. This includes requests that join and leave mid-run, contexts up to 229,376 tokens, tool calls, images, cancellation and engine failures.
* A 60-minute run with 10,178 mixed requests (text, tools and images, fastest model, N = 8): no difference, nothing stuck, and memory did not keep growing.

Not tested:

* Other GPUs, including 24 GB and 16 GB cards. On them the start-up check decides.
* Speed with the quality model, and with thinking on.
* Linux with `--parallel`.
* Two GPUs.

Known limits:

* The 16-row pass is shared, so each request is slower than alone.
* Reading a prompt pauses all others. Reading 66,416 tokens takes about 20 seconds.
* The next turn after tool calls must come within the kept time (1 second for a short conversation). Later, it reads the conversation again. The answer is the same.
* The wait limit counts only waiting caused by kept places and next turns. A next turn that runs alone (because no place is free) is not stopped for a new request, so a new request can wait longer than the limit.
* Over long runs the GPU memory of the whole card rose by up to 71 MiB. In the last 30-minute run the server's own GPU memory did not change.

## Can I use other models, for example Qwen4?

Not in this release: it runs Qwen3.8-27B only. The program that made the two model files ([tools/convert/](../tools/convert/), added on
November 14, 2026) knows the layers of Qwen3.8-27B only, and the engine runs this model only, so another model needs changes in both.
Whether other models can be supported is decided one by one, and nothing is promised.

## Does it send my prompts anywhere?

No. The Python package makes no outgoing network connections (it only listens, on `127.0.0.1` unless you choose otherwise), and the engine library
imports no networking functions. After the download it needs no internet connection. The one exception is the update check: `yasakani update`
when you type it, and the quickstart's check for a new version, each send one request to GitHub that holds no prompt and nothing about your PC
([below](#how-does-the-update-work)). Details and the checks behind this statement are in [SECURITY.md](../SECURITY.md).

## How does the update work?

Since 0.2.0, the quickstart (`START.cmd`) can update itself. From v0.1.6 or older, download the 0.2.0 quickstart ZIP by hand once and run `START.cmd`; the models are kept.

* **The check.** At most once a day, when the menu opens, the quickstart first prints that it asks github.com, then sends one HTTPS request:
  `GET https://api.github.com/repos/yasakani-dev/yasakani/releases/latest`, with the User-Agent `yasakani-quickstart` and nothing else about
  your PC (no ID, no version of Windows, no GPU, no prompt). It waits at most 5 seconds; if GitHub cannot be reached, it says so in one line and
  goes on. Pre-releases are never offered. GitHub sees your IP address, as with any download.
* **Turning it off.** `START.cmd --no-update-check` turns the daily check off and remembers it (in `quickstart_state.json`);
  `START.cmd --update-check` turns it on again. The environment variable `YASAKANI_NO_UPDATE_CHECK=1` also turns it off. The menu item
  “check for a new version” and `START.cmd --action update` still work when you choose them.
* **Nothing changes without a yes.** When there is a newer version, the quickstart shows the old and new version and the release page, and
  asks. In the browser chat, a bar offers **Update** and **Later**; the page itself sends nothing to GitHub.
* **What an update does.** It downloads the quickstart ZIP and `SHA256SUMS.txt` of that release, only from
  `https://github.com/yasakani-dev/yasakani/releases/download/<tag>/`, compares the ZIP's SHA-256 with `SHA256SUMS.txt`, and accepts only
  the known start files (no folders, no other files). It copies the current start files to `backup\<old version>\`, puts the new ones in
  place and starts the new quickstart, which downloads its own engine ZIP and checks its SHA-256 as on the first run, then installs the
  Python package again. The models are kept.
* **Going back.** `START.cmd --action rollback` puts the start files of the version before the last update back; on the next start the
  engine of that version is installed again.
* **How far you can trust it.** The release files are not code-signed. The check rests on GitHub's HTTPS and on the release's
  `SHA256SUMS.txt`, which comes from the same release: it catches a broken or swapped download, not a release that was itself replaced on
  GitHub.
* **`yasakani update`** depends on how YASAKANI was installed. In a folder set up by the quickstart it asks GitHub nothing itself: it
  opens a new window that runs `START.cmd --action update` and ends; the quickstart then checks, asks and replaces the files (`--yes`
  skips the question). After a manual installation it asks GitHub once, prints the installed and the latest version, prints what to
  download and the command to run, and changes nothing. `--check` only prints, in both cases. `yasakani serve`, `chat` and `bench` never contact the network.

## Windows warns me about the download. Is that normal?

The release files are not code-signed, so Windows SmartScreen or an antivirus program may warn about them. Compare the SHA-256 checksum of what you
downloaded with the one on the release page, and run `yasakani verify` on the model folder. You can also build `yskn.dll` from the source.

## Can I use it from my own programs?

Yes: it serves the OpenAI chat-completions API on `http://127.0.0.1:8765/v1`. Supported request fields, the way the thinking text is returned and the
loop guard are described in [python/README.md](../python/README.md). It generates one request at a time; `n` above 1
and `logprobs` are not supported. Tool / function calling is supported since 0.1.1 (`tools` with `tool_choice` `auto` or `none`; since 0.1.4 also `required` or a named
function, accepted but not enforced).
Images are accepted since 0.1.2 through the optional image input (a community contribution, [#4](https://github.com/yasakani-dev/yasakani/pull/4); on Linux with llama.cpp
built from source, contributed in [#8](https://github.com/yasakani-dev/yasakani/pull/8) and not tried by us; see
[python/README.md](../python/README.md#image-input-optional)).

## Can I use it commercially?

YASAKANI is under the Apache License 2.0, which allows commercial use; Qwen3.8-27B (Alibaba Cloud) and the helper model are published under Apache-2.0
as well. The licenses, copyright notices and what was converted or re-trained are in [LICENSE](../LICENSE), [NOTICE](../NOTICE) and
[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md). Read them for your own case; this FAQ is not legal advice.

## Who made the model, and what did you change?

Qwen3.8-27B is the work of Alibaba's Qwen team. YASAKANI is the engine and the file format that run it fast, and the model files are converted,
compressed versions of it. The small helper model that speeds up generation is built on a publicly released model: it was converted, compressed and
re-trained. [NOTICE](../NOTICE) says where each comes from.

## How do I report a problem?

See [CONTRIBUTING.md](../CONTRIBUTING.md) for bugs and test results, and [SECURITY.md](../SECURITY.md) for security problems.
