# Thinking off: YASAKANI measured again with 0.1.4 (November 15, 2026)

Output of `tools/speed/tables_0114.py`. YASAKANI's rows come from `bench/20261115` (0.1.4, measured on November 15 with the
clients of `bench/20261108`: thinking off, greedy, 2 runs per task); the other engines are the records of `bench/20261108`
(November 3 and 6). The rule for runs that are left out is the one of `tools/speed/tables_strict.py`, the same for every engine.
tok/s; conditions and limits as in the README of `bench/`.

## The table of the README (average without repetition loops and unrequested tool calls)

| Engine | Code | Story | Translation | Structured | Code edit | JSONL edit | Long JSONL |
|---|---|---|---|---|---|---|---|
| YASAKANI fastest model (0.1.4) | 512 | 195 | 479 | 882 | 2,025 | 2,298 | 1,214 |
| YASAKANI quality model (0.1.4) | 468 | 172 | 464 | 892 | 2,008 | 2,243 | 1,115 |
| MegaCapybara 1.39 Tiny | 465 | 164 | 268 | 599 | 441 | 658 | 666 |
| mkyt v0.3.2 | 395 | 181 | 374 | 555 | 423 | 634 | 639 |
| ignis v0.9.6 | 330 | 167 | 298 | 439 | 331 | 510 | 514 |
| NInfer (November 2 build) | 264 | 123 | 257 | 386 | 295 | 443 | 438 |
| vLLM 0.31.0 | 290 | 148 | 261 | 354 | 241 | 426 | 431 |
| SGLang 0.5.22 | 252 | 129 | 246 | 371 | 430 | 466 | 451 |
| llama.cpp b11700 | 242 | 133 | 232 | 337 | 244 | 360 | 362 |
| TensorFold 0.6.6 | 181 | 104 | 172 | 405 | 383 | 408 | 261 |
| (largest value of the other engines) | 465 | 181 | 374 | 599 | 441 | 658 | 666 |
| YASAKANI fastest model (0.1.4) / largest value of the others | 1.10 | 1.08 | 1.28 | 1.47 | 4.59 | 3.49 | 1.82 |
| YASAKANI quality model (0.1.4) / largest value of the others | 1.01 | 0.95 | 1.24 | 1.49 | 4.55 | 3.41 | 1.67 |

## Average of all runs (no run left out)

| Engine | Code | Story | Translation | Structured | Code edit | JSONL edit | Long JSONL |
|---|---|---|---|---|---|---|---|
| YASAKANI fastest model (0.1.4) | 1,089 | 281 | 479 | 882 | 2,025 | 2,298 | 1,214 |
| YASAKANI quality model (0.1.4) | 468 | 262 | 464 | 892 | 2,008 | 2,243 | 1,115 |
| MegaCapybara 1.39 Tiny | 465 | 164 | 268 | 599 | 441 | 658 | 666 |
| mkyt v0.3.2 | 438 | 204 | 374 | 555 | 423 | 634 | 639 |
| ignis v0.9.6 | 330 | 167 | 298 | 439 | 331 | 510 | 514 |
| NInfer (November 2 build) | 282 | 123 | 257 | 386 | 295 | 443 | 438 |
| vLLM 0.31.0 | 312 | 150 | 261 | 354 | 241 | 426 | 431 |
| SGLang 0.5.22 | 270 | 129 | 246 | 371 | 430 | 466 | 451 |
| llama.cpp b11700 | 242 | 143 | 232 | 337 | 244 | 360 | 362 |
| TensorFold 0.6.6 | 181 | 104 | 172 | 405 | 383 | 408 | 261 |

## 0.1.2, 0.1.3 and 0.1.4 one after the other

The same clients, the same PC and model files, one server at a time (fastest model 0.1.2 -> 0.1.3 -> 0.1.4, then quality model
0.1.4 -> 0.1.3 -> 0.1.2). The last column is the extra task of `bench/20261115/prompts/cases_plus.json`.

### YASAKANI fastest model

| Version | Code | Story | Translation | Structured | Code edit | JSONL edit | Long JSONL | Long JSONL edit |
|---|---|---|---|---|---|---|---|---|
| 0.1.2 | 515 | 195 | 499 | 911 | 1,210 | 1,125 | 1,218 | 1,141 |
| 0.1.3 | 514 | 195 | 479 | 878 | 1,188 | 1,124 | 1,221 | 1,137 |
| 0.1.4 | 512 | 195 | 479 | 882 | 2,025 | 2,298 | 1,214 | 2,099 |
| 0.1.4 / 0.1.2 | 0.99 | 1.00 | 0.96 | 0.97 | 1.67 | 2.04 | 1.00 | 1.84 |
| 0.1.3 / 0.1.2 | 1.00 | 1.00 | 0.96 | 0.96 | 0.98 | 1.00 | 1.00 | 1.00 |

Runs left out: 0.1.2: scenario_code_cuda r0 L, scenario_code_cuda r1 L, scenario_code_typescript r0 T, scenario_code_typescript r1 T, scenario_story_en_mystery r0 L, scenario_story_en_mystery r1 L; 0.1.3: scenario_code_cuda r0 L, scenario_code_cuda r1 L, scenario_code_typescript r0 T, scenario_code_typescript r1 T, scenario_story_en_mystery r0 L, scenario_story_en_mystery r1 L; 0.1.4: scenario_code_cuda r0 L, scenario_code_cuda r1 L, scenario_code_typescript r0 T, scenario_code_typescript r1 T, scenario_story_en_mystery r0 L, scenario_story_en_mystery r1 L

Edit answers equal to the expected file (`exact`): 0.1.2: edit_code r0 yes, edit_code r1 yes, edit_jsonl r0 yes, edit_jsonl r1 yes, edit_jsonl_long r0 yes, edit_jsonl_long r1 yes; 0.1.3: edit_code r0 yes, edit_code r1 yes, edit_jsonl r0 yes, edit_jsonl r1 yes, edit_jsonl_long r0 yes, edit_jsonl_long r1 yes; 0.1.4: edit_code r0 yes, edit_code r1 yes, edit_jsonl r0 yes, edit_jsonl r1 yes, edit_jsonl_long r0 yes, edit_jsonl_long r1 yes

| Prompt (12 published) | 0.1.2 | 0.1.3 | 0.1.4 | 0.1.4 / 0.1.2 |
|---|---|---|---|---|
| code_cuda | left out (LL) | left out (LL) | left out (LL) | - |
| code_python | 515 | 514 | 512 | 0.994 |
| code_typescript | left out (TT) | left out (TT) | left out (TT) | - |
| story_en_mystery | left out (LL) | left out (LL) | left out (LL) | - |
| story_zh_dialogue | 185 | 186 | 186 | 1.003 |
| story_zh_scifi | 204 | 204 | 204 | 0.999 |
| structured_csv | 817 | 789 | 788 | 0.965 |
| structured_jsonl | 1,218 | 1,170 | 1,182 | 0.970 |
| structured_sql | 699 | 674 | 675 | 0.966 |
| translation_en_zh | 509 | 487 | 487 | 0.958 |
| translation_markdown | 375 | 363 | 360 | 0.959 |
| translation_zh_en | 612 | 587 | 591 | 0.965 |

### YASAKANI quality model

| Version | Code | Story | Translation | Structured | Code edit | JSONL edit | Long JSONL | Long JSONL edit |
|---|---|---|---|---|---|---|---|---|
| 0.1.2 | 464 | 175 | 478 | 902 | 1,128 | 1,146 | 1,155 | 1,096 |
| 0.1.3 | 468 | 172 | 465 | 893 | 1,089 | 1,103 | 1,117 | 1,059 |
| 0.1.4 | 468 | 172 | 464 | 892 | 2,008 | 2,243 | 1,115 | 2,056 |
| 0.1.4 / 0.1.2 | 1.01 | 0.98 | 0.97 | 0.99 | 1.78 | 1.96 | 0.97 | 1.88 |
| 0.1.3 / 0.1.2 | 1.01 | 0.98 | 0.97 | 0.99 | 0.97 | 0.96 | 0.97 | 0.97 |

Runs left out: 0.1.2: scenario_story_en_mystery r0 L, scenario_story_en_mystery r1 L; 0.1.3: scenario_story_en_mystery r0 L, scenario_story_en_mystery r1 L; 0.1.4: scenario_story_en_mystery r0 L, scenario_story_en_mystery r1 L

Edit answers equal to the expected file (`exact`): 0.1.2: edit_code r0 no, edit_code r1 no, edit_jsonl r0 yes, edit_jsonl r1 yes, edit_jsonl_long r0 yes, edit_jsonl_long r1 yes; 0.1.3: edit_code r0 no, edit_code r1 no, edit_jsonl r0 yes, edit_jsonl r1 yes, edit_jsonl_long r0 yes, edit_jsonl_long r1 yes; 0.1.4: edit_code r0 no, edit_code r1 no, edit_jsonl r0 yes, edit_jsonl r1 yes, edit_jsonl_long r0 yes, edit_jsonl_long r1 yes

| Prompt (12 published) | 0.1.2 | 0.1.3 | 0.1.4 | 0.1.4 / 0.1.2 |
|---|---|---|---|---|
| code_cuda | 457 | 458 | 457 | 1.000 |
| code_python | 481 | 495 | 495 | 1.030 |
| code_typescript | 455 | 450 | 451 | 0.990 |
| story_en_mystery | left out (LL) | left out (LL) | left out (LL) | - |
| story_zh_dialogue | 172 | 169 | 169 | 0.985 |
| story_zh_scifi | 179 | 174 | 174 | 0.977 |
| structured_csv | 915 | 885 | 885 | 0.968 |
| structured_jsonl | 1,057 | 1,087 | 1,085 | 1.026 |
| structured_sql | 734 | 708 | 705 | 0.961 |
| translation_en_zh | 471 | 460 | 459 | 0.974 |
| translation_markdown | 372 | 359 | 363 | 0.976 |
| translation_zh_en | 591 | 574 | 571 | 0.966 |

The thinking-on tables were not measured again; they are those of `bench/20261108/tables/SPEED_TABLES.md` (the engine of
0.1.0 to 0.1.2, the build of November 7).
