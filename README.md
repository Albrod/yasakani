# YASAKANI

English · [日本語](README.ja.md)

<a name="what-it-is"></a>

YASAKANI runs Qwen3.8-27B on a Windows PC with one NVIDIA RTX GPU. The developer has tested it on the RTX 5090 only. After setup you can chat and edit code offline, or connect your development tools to its OpenAI-compatible API.

**[Download for Windows](https://github.com/yasakani-dev/yasakani/releases/download/v0.1.6/yasakani-quickstart-0.1.6.zip)** · [Compare eight engines](docs/MEASUREMENTS.md#thinking-off-the-table-of-the-readme-average-without-repetition-loops-etc-toks) · [Watch an uncut code edit](https://github.com/yasakani-dev/yasakani/releases/download/v0.1.6/fast_live_uncut.mp4)

<a name="two-models"></a>
<a name="what-you-need"></a>
<a name="limits"></a>

## Models and hardware

![One RTX 5090, one request at a time, thinking off. On our own JSONL edit task (return 60 JSONL lines with one record changed) the recommended quality model (12.37 GB) wrote 2,243 tok/s and the fastest model (10.31 GB) 2,298 tok/s. v0.1.4, November 15, 2026, mean of 2 runs, prompt reading excluded. Context up to 262,144 tokens.](docs/assets/hero-en.svg)

JSONL edit is our own task: the model returns 60 JSONL lines with one record changed, so almost all of the answer copies the input. One RTX 5090, one request at a time, thinking off, v0.1.4 (November 15, 2026), mean of 2 runs, prompt reading excluded. Other GPUs are slower.

Choose the **quality model** for everyday use and predictions closest to the original model. Choose the **fastest model** for its smaller file and higher speed in most tasks. [Quality model download](https://huggingface.co/yasakani-dev/Qwen3.8-27B-YASAKANI-quality) · [Fastest model download](https://huggingface.co/yasakani-dev/Qwen3.8-27B-YASAKANI-speed)

| Your GPU | Model choice | Windows testing |
|---|---|---|
| RTX 5090 (32 GB) | Quality or fastest | Tested by the developer |
| RTX 3090, 3090 Ti, or 4090 (24 GB) | Quality or fastest | Built for these cards; not tested or tuned by the developer |
| Listed RTX 40 and 50 series cards (16 GB) | Fastest, with context sized to free memory | Built for these cards; not tested or tuned by the developer |

All speed numbers in this repository are from the RTX 5090. Other cards are slower, and their speed and operation are not guaranteed. The wide copy passes behind the edit numbers are used on compute capability 12.x (RTX 50 series) only, and were measured on the RTX 5090 only. RTX 30 and 40 cards check at most 16 copied tokens per pass, so the edit speedup of 0.1.4 does not apply to them.

Contributors have tested a Linux build from source. AMD and Mac are future targets. [GPU list and requirements](docs/FAQ.md#which-gpus-which-systems) · [Build on Linux](BUILDING.md#linux-building-from-source)

**Several GPUs:** one process uses one GPU. `--device N` picks GPU N as `yasakani info` and nvidia-smi number them. With 0.1.5, `--device 1` on a PC with two GPUs could stop with `graph instantiate failed` ([#20](https://github.com/yasakani-dev/yasakani/issues/20)); 0.1.6 has a fix for this, but it has not yet been tried on a PC with two GPUs.

## Quick start

You need 64-bit Windows 11, Python 3.10 or later, NVIDIA driver 580 or later, the Visual C++ 2015–2022 x64 runtime, and 20 GB of free disk space.

1. Extract the v0.1.6 Windows quickstart ZIP to a path with only English letters and digits, such as `C:\yasakani`. [Release files and SHA-256 checksums](https://github.com/yasakani-dev/yasakani/releases/tag/v0.1.6)
2. Open **`START.cmd`** and choose a model. The launcher downloads the engine and model files, verifies them, and runs the GPU self-check.
3. Select browser or terminal chat and enter your first prompt. To use another app, start the server and point the client at `http://127.0.0.1:8765/v1`.

In terminal chat, enter `/effort medium` for a shorter thinking phase or `/think off` to turn it off. [Manual installation](python/README.md#install) · [API guide](python/README.md#the-server)

## Community conversions

Files that other people made with the converter in [tools/convert/](tools/convert/). They are **not checked or maintained by this
project**; the person named below keeps them working, and a conversion may be removed from this list in a release where it has fallen
behind.

**Arete-27B** (by [@audreyt](https://github.com/audreyt)): a community conversion of [NousResearch/Arete-27B](https://huggingface.co/NousResearch/Arete-27B),
Nous Research's fine-tune of Qwen3.8-27B, made with the quality model's settings and reusing its draft model unchanged. Files:
[audreyt/Arete-27B-YASAKANI-quality](https://huggingface.co/audreyt/Arete-27B-YASAKANI-quality/tree/659effbdf9dc103fcc053e3da7858bcff7f14cc9) at revision
`659effb`; the SHA-256 of every file is in its `SHA256SUMS` (model file `arete_quality.yskn`, 12,365,242,368 bytes,
`541c9206ede621bd92e78ff09f838fca5e4660d834fa883b3d20cdb4ab56880a`).

```bat
hf download audreyt/Arete-27B-YASAKANI-quality --revision 659effbdf9dc103fcc053e3da7858bcff7f14cc9 --local-dir D:\models\arete-quality
yasakani selfcheck --model-dir D:\models\arete-quality --dll D:\yasakani\yskn.dll
```

Then use it as in [Quick start](#quick-start) with `--model-dir D:\models\arete-quality`. Measured by the author of this conversion on an
RTX 5090:

* Closeness to its own BF16 original (`tools/quality/`, 20 thinking windows, KV5 cache, measured with 0.1.3): KL divergence 0.0116,
  same most likely token 95.64%. The quality model, measured the same way against Qwen3.8-27B: 0.0126 and 95.35%.
* With 0.1.4: `yasakani verify` 11 of 11 files, `yasakani selfcheck` 3 of 3 (the same outputs as with 0.1.3). Tokens per look-ahead
  pass: prose 2.40, thinking 2.48, code 4.68, CSV to JSON 15.04. On thinking text that is fewer than the quality model (2.48 against
  3.24 in the author's run of both on the same card with 0.1.3), so expect it to be slower there than the speed tables above.
* Further evaluations, with their settings and caveats, are on the [model card](https://huggingface.co/audreyt/Arete-27B-YASAKANI-quality/tree/659effbdf9dc103fcc053e3da7858bcff7f14cc9).

<a name="speed"></a>

## Speed and quality

All engines generated one request at a time on the same PC with one stock RTX 5090 (600 W). The fastest model led the eight other engines in all seven categories, using filtered means with thinking off and the best run per task with thinking on. The big gaps are in our own three tasks: two edits whose answers mostly copy the input, and one long, very regular JSONL output. With thinking off they are 1.8 to 4.6 times the next engine (code edit: 2,025 tok/s with v0.1.4 on November 15, against 441 tok/s for MegaCapybara 1.39 Tiny on November 3). On the 12 published prompts the fastest model is 1.1 to 1.5 times the next engine. Three of those 12 (CUDA code, TypeScript code, the English mystery story) are not compared, because its answer fell into a repetition loop or made an unrequested tool call.

<a name="what-014-changed"></a>

**What 0.1.4 changed** (thinking off, 2 runs per task, measured one after the other against 0.1.2 on November 15, 2026; RTX 5090; tok/s):

| | Code edit | JSONL edit | Long JSONL edit (160 records) | Other tasks |
|---|---|---|---|---|
| fastest model | 1,210 → 2,025 (1.67×) | 1,125 → 2,298 (2.04×) | 1,141 → 2,099 (1.84×) | 0.96 to 1.00× |
| quality model | 1,128 → 2,008 (1.78×) | 1,146 → 2,243 (1.96×) | 1,096 → 2,056 (1.88×) | 0.97 to 1.01× |

The answers were the same text, except one 0.1.2 run that changed part-way (the rare change fixed in 0.1.3). 0.1.5 and 0.1.6 have the same GPU code. [Details](docs/MEASUREMENTS.md#speed-thinking-off)

To measure your own PC, run `yasakani bench`. `yasakani bench --edit` runs the three edit tasks above (the code edit, the JSONL edit and the long JSONL edit) and prints the speed, the tokens per pass and whether each answer is the expected one. [How to measure the edit speed](docs/FAQ.md#how-do-i-measure-the-edit-speed-on-my-pc)

<details>
<summary>See all seven categories with thinking off and on</summary>

**Thinking off** uses the two-run mean for each task, excluding repetition loops and unrequested tool calls by the same rule for all engines. YASAKANI was measured again on November 15 with v0.1.4 (0.1.5 and 0.1.6 have the same GPU code). Numbers are generation tok/s, excluding prompt processing.

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

**Thinking on** takes the fastest run per task and averages within each category. YASAKANI was measured on November 8 with the engine of 0.1.0 to 0.1.2 (the build of November 7), using five runs per task; it was not measured again with 0.1.4. The other engines had two.

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

The other engines were measured on November 3 and 6. Each engine used its own weights, quantization, KV cache and acceleration settings. [Measurement conditions and all-run averages](docs/MEASUREMENTS.md) · [Records and reproduction scripts](bench/README.md)

</details>

<a name="one-edit-start-to-finish"></a>

In the uncut November 8 recording, the fastest model edited a Python file and returned the full file **5.56 seconds** after submission. The edited code matched the original program’s behavior on 17,701 test inputs. [Recording and verification records](bench/20261108/demo/)

<a name="quality"></a>

On 20 held-out thinking texts (20,460 positions), the quality model chose the same most likely next token as the original BF16 model at **95.24%** of positions (KL 0.0122), and the fastest model at **93.01%** (KL 0.0304). The similarly sized GSQ-RCO IQ3_S agreed at 93.64%. These were measured on November 8 with a research build and an FP16 KV cache. Measured again on November 14 with the released engine (0.1.0 to 0.1.2) and its KV5 cache: 95.35% (KL 0.0126) and 92.98% (KL 0.0309). This is agreement with the original model on thinking text, not task accuracy, and the models are not identical to the original. [Quality results for 13 configurations and methodology](docs/MEASUREMENTS.md#quality-on-thinking-text) · [Measure it yourself](tools/quality/)

<a name="long-contexts"></a>

With the maximum context of 262,144 tokens reserved, GPU memory use including the helper model was about 21.2 GiB for the quality model and 19.1 GiB for the fastest. Long-context quality was measured up to 128K tokens. [KV5 measurements](docs/MEASUREMENTS.md#kv5-the-compressed-key-value-cache-with-the-fastest-model)

<a name="how-it-works-briefly"></a>
<a name="documentation-and-contributions"></a>

## Development and support

YASAKANI’s weight format and GPU kernels were designed together. During edits the main model can verify spans copied from the input in one pass, which cuts the time spent writing unchanged lines. [Technical notes](docs/TECHNOLOGY.md) · [File format](docs/FORMAT.md) · [Conversion tools](tools/convert/)

<a name="use-it-as-an-agent"></a>

Build from source, report a result from your GPU, or try the optional browser agent, which needs 24 GB of GPU memory and bundles an older server. [Build guide](BUILDING.md) · [Contribute](CONTRIBUTING.md) · [Agent setup](harness/HARNESS_README.txt)

<a name="support"></a>

This engine was developed in Japan. Support goes toward test hardware and development time, starting with 24 GB and 16 GB RTX cards. Backers who opt in are credited in the repository and in the startup credits of a later release. [CAMPFIRE for backers in Japan](https://camp-fire.jp/projects/992233/view) · [GitHub Sponsors](https://github.com/sponsors/yasakani-dev)

<a name="license"></a>

Released under Apache License 2.0. [License](LICENSE) · [Copyright notices](NOTICE) · [Third-party notices](THIRD_PARTY_NOTICES.md) · [Release history and contributors](docs/RELEASE_NOTES.md)
