# -*- coding: utf-8 -*-
"""YASAKANI quickstart (Windows + one NVIDIA RTX GPU of the RTX 30, 40 or 50 series with 16 GB or more; tested on the RTX 5090).

Double-click START.cmd.  This script then
  1. checks this computer (Windows, Python, GPU, NVIDIA driver, free disk space) before anything is downloaded,
  2. lets you pick one of the two models (on a GPU with less than 24 GB only the fastest model fits),
  3. downloads the engine zip from the release page and checks its SHA-256,
  4. makes a private Python environment in this folder and installs the package,
  5. downloads the model folder from Hugging Face,
  6. runs `yasakani info`, `yasakani verify` and `yasakani selfcheck`,
  7. and offers: measure the speed, compare look-ahead on and off, chat in your browser or in the terminal, or start the
     local server.

The browser chat (chat.html) is served by this script on 127.0.0.1 only.  The script passes the requests of that page on to
`yasakani serve`, which runs on this computer too.  Nothing is sent anywhere else.

Updates: at most once a day, when the menu is shown, the script asks GitHub's API for the latest release (one HTTPS request
without anything about this computer; it says so first, and --no-update-check turns it off).  Nothing is updated without a yes.  An
update downloads the start files of that release, checks them against its SHA256SUMS.txt, keeps the current ones in backup/<version>/,
puts the new ones in place and runs the new quickstart.py, which installs its own engine as usual.  --action rollback goes back.

Everything is kept inside the folder this file is in.  Nothing is installed system-wide.
Run it again any time: finished steps are skipped.

Apache License 2.0 - part of https://github.com/yasakani-dev/yasakani
"""
import argparse
import hashlib
import hmac
import http.client
import io
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

QS_VERSION = "0.2.0"
ENGINE_ZIP = "yasakani-0.2.0-win-x64.zip"
ENGINE_URL = "https://github.com/yasakani-dev/yasakani/releases/download/v0.2.0/" + ENGINE_ZIP
ENGINE_SHA256 = "0d820dd5eafc700c5b8ea7cec093efbd179cf831a0c65b64947c6eccb2d39d42"
REPO_URL = "https://github.com/yasakani-dev/yasakani"
SUPPORT_URL = "https://camp-fire.jp/projects/992233/view"
ENGINE_PORT, WEB_PORT = 8765, 8766      # `yasakani serve` and the browser chat; the next free ports are used if these are taken
MAX_BODY = 16 << 20

# ---- updates.  The check is one GET of the API below (5 s, no identifiers); the files come only from this repository's release
# download URLs; the start files are the members of the quickstart zip (release.py: LICENSE + quickstart/*).
UPDATE_API = "https://api.github.com/repos/yasakani-dev/yasakani/releases/latest"
UPDATE_DOWNLOAD = REPO_URL + "/releases/download/"
UPDATE_UA = "yasakani-quickstart"
START_FILES = ("LICENSE", "README.txt", "START.cmd", "chat.html", "lookahead_check.py", "quickstart.py")
MAX_START_FILE, MAX_START_ZIP, MAX_SUMS, MAX_API = 4 << 20, 16 << 20, 64 << 10, 1 << 20
CHECK_EVERY = 24 * 3600
UPDATE_RC = 75                      # run_web: the page asked for the update; the server is stopped
PAGE_CSP = ("default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'; "
            "base-uri 'none'; form-action 'none'")
MODELS = {
    "quality": {"repo": "yasakani-dev/Qwen3.8-27B-YASAKANI-quality", "need_gb": 16},
    "speed": {"repo": "yasakani-dev/Qwen3.8-27B-YASAKANI-speed", "need_gb": 14},
}

# ---- GPUs (0.1.3).  The engine is built for compute capability 8.6, 8.9 and 12.x (RTX 30, 40 and 50 series) and tested on the
# RTX 5090 only.  The context rule is the one of yasakani/gpu.py, which the package applies each time it opens the model; it is
# repeated here only to tell the context before anything is downloaded (python/tests/test_gpu.py checks that the two agree).
GPU_FIELDS = ("index", "uuid", "name", "compute_cap", "memory.total", "memory.used", "driver_version")
MIN_TOTAL_MIB = 15000                 # "16 GB": the smallest cards the models fit on (fastest model only)
QUALITY_MIN_TOTAL_MIB = 23000         # "24 GB": the quality model needs this much
FULL_CONTEXT_MIN_TOTAL_MIB = 30000    # cards with this much memory keep the full context (the RTX 5090)
KV_BYTES_PER_TOKEN, CONTEXT_STEP, MIN_CONTEXT, HEAD_OFF_BELOW, MARGIN_MIB, FULL_CONTEXT = 22400, 16384, 16384, 65536, 512, 262144
MODEL_BYTES = {   # weights, draft model, its output head (the published files)
    "quality": (12365242368, 1164969472, 1411010584),
    "speed": (10311344128, 1164969472, 1411010584),
}
OVERHEAD_MIB = {"quality": 2300, "speed": 2100}

# ---- the NVIDIA driver (0.1.4).  yskn.dll carries the CUDA 13.2 runtime (linked in, there is no separate runtime DLL).  Under CUDA minor
# version compatibility a program built with any CUDA 13.x toolkit runs on a driver of the R580 branch or newer, which reports CUDA 13.0
# or newer (NVIDIA, CUDA Toolkit release notes, "Minimum Required Driver Version for CUDA Minor Version Compatibility": 13.x >= 580;
# https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html).  With an older driver the engine stops at its first
# CUDA call, which comes after the engine and the model have been downloaded, so the quickstart checks the driver in step 1.
MIN_DRIVER = 580                  # the first number of nvidia-smi's driver_version: 580.88 or 616.92 (Windows), 580.65.06 (Linux)
MIN_CUDA_DRIVER_API = 13000       # cuDriverGetVersion() of such a driver: 13000 = CUDA 13.0 (driver 616.92 gives 13040 = CUDA 13.4)

T = {
    "banner": {
        "ja": "YASAKANI クイックスタート {v}  -  Qwen3.8-27B を NVIDIA の RTX 1 枚で動かします（確かめたのは RTX 5090）",
        "en": "YASAKANI quickstart {v}  -  runs Qwen3.8-27B on one NVIDIA RTX card (tested on the RTX 5090)",
    },
    "support": {
        "ja": "開発を応援してください: CAMPFIRE のクラウドファンディング（2026 年 12 月 16 日まで）\n  {u}",
        "en": "Like it? A star on GitHub helps: {r}\n  (The crowdfunding at {u} is open to residents of Japan.)",
    },
    "not_windows": {
        "ja": "このソフトは Windows 11 専用です。",
        "en": "This software runs on Windows 11 only.",
    },
    "old_python": {
        "ja": "Python 3.10 以上が必要です（いまは {v}）。",
        "en": "Python 3.10 or newer is needed (this is {v}).",
    },
    "bad_path": {
        "ja": "このフォルダーの場所に、日本語などの英数字以外の文字が入っています:\n  {p}\n"
              "C:\\yasakani のような英数字だけの場所にフォルダーごと移して、もう一度 START.cmd をダブルクリックしてください。",
        "en": "The path of this folder contains characters outside plain ASCII:\n  {p}\n"
              "Move the folder to a plain path such as C:\\yasakani and double-click START.cmd again.",
    },
    "step": {"ja": "[{i}/{n}] {t}", "en": "[{i}/{n}] {t}"},
    "s_check": {"ja": "このパソコンの確認", "en": "Checking this computer"},
    "s_model": {"ja": "モデルを選ぶ", "en": "Choosing the model"},
    "s_engine": {"ja": "エンジンのダウンロードと確認", "en": "Downloading and checking the engine"},
    "s_venv": {"ja": "Python の環境を作って、パッケージを入れる", "en": "Making the Python environment and installing the package"},
    "s_dl": {"ja": "モデルのダウンロード", "en": "Downloading the model"},
    "s_verify": {"ja": "点検（info・verify・selfcheck）", "en": "Checks (info, verify, selfcheck)"},
    "s_menu": {"ja": "準備ができました", "en": "Ready"},
    "no_smi": {
        "ja": "nvidia-smi が見つかりません。NVIDIA の GPU とドライバーが要ります。ドライバーを入れてから、もう一度試してください。\n"
              "AMD・Intel の GPU と Mac には対応していません。",
        "en": "nvidia-smi was not found. YASAKANI needs an NVIDIA GPU and its driver; install the driver and try again.\n"
              "AMD and Intel GPUs and Macs are not supported.",
    },
    "no_gpu": {
        "ja": "NVIDIA の GPU が見つかりません。YASAKANI は NVIDIA の RTX 30・40・50 シリーズ（GPU のメモリ 16 GB 以上）向けに作ってあります（確かめたのは RTX 5090 だけ）。\n"
              "AMD・Intel の GPU と Mac には対応していません。",
        "en": "No NVIDIA GPU was found. YASAKANI is built for NVIDIA RTX 30, 40 and 50 series cards with 16 GB or more (tested on the RTX 5090 only).\n"
              "AMD and Intel GPUs and Macs are not supported.",
    },
    "gpu_is": {
        "ja": "GPU: {g}（計算能力 {c}、メモリ {t} MiB）",
        "en": "GPU: {g} (compute capability {c}, {t} MiB)",
    },
    "gpu_pick": {"ja": "GPU が {n} 枚あります。使うのは {g} です。", "en": "There are {n} GPUs. Using {g}."},
    "gpu_tested": {"ja": "動作を確かめてある GPU です。", "en": "This is the GPU YASAKANI is tested on."},
    "gpu_other": {
        "ja": "この GPU は動くように作ってありますが、最適化も動作の確認もしていません（確かめたのは RTX 5090 だけです）。\n"
              "速さは RTX 5090 より遅くなる見込みで、動作も保証しません。",
        "en": "YASAKANI is made to run on this GPU, but it has not been optimized or tested on it (only the RTX 5090 has been tested).\n"
              "It is expected to be slower than on the RTX 5090, and its operation is not guaranteed.",
    },
    "gpu_small": {
        "ja": "この GPU のメモリ（{t} MiB）では足りません。YASAKANI には GPU のメモリが 16 GB 以上要ります。",
        "en": "This GPU's memory ({t} MiB) is too small. YASAKANI needs a GPU with 16 GB or more.",
    },
    "gpu_old": {
        "ja": "この GPU（計算能力 {c}）では動きません。動くように作ってあるのは RTX 30・40・50 シリーズ（計算能力 8.6 以上）で、メモリが 16 GB 以上のカードです。",
        "en": "YASAKANI does not run on this GPU (compute capability {c}). It is built for RTX 30, 40 and 50 series cards "
              "(compute capability 8.6 or higher) with 16 GB or more.",
    },
    "gpu_nocode": {
        "ja": "この GPU（計算能力 {c}）用のコードは、この版のエンジンに入っていません。動くのは RTX 30・40・50 シリーズ"
              "（計算能力 8.6・8.9・12.x）です。",
        "en": "The engine of this version has no code for this GPU (compute capability {c}). It runs on RTX 30, 40 and 50 series "
              "cards (compute capability 8.6, 8.9 and 12.x).",
    },
    "driver_ok": {
        "ja": "ドライバー: {d}（580 以上なので使えます）",
        "en": "Driver: {d} (580 or newer: fine)",
    },
    "driver_old": {
        "ja": "NVIDIA のドライバーが古いので止めました（今は {d}）。エンジンは CUDA 13 で作ってあり、ドライバー 580 以上が要ります。\n"
              "新しいドライバーを入れてパソコンを再起動してから、もう一度 START.cmd をダブルクリックしてください。\n"
              "  https://www.nvidia.com/ja-jp/drivers/\n"
              "（エンジンとモデルのダウンロードの前に確かめています）",
        "en": "Stopped: the NVIDIA driver is too old ({d}). The engine is built with CUDA 13 and needs driver 580 or newer.\n"
              "Install a current driver, restart the PC and double-click START.cmd again.\n"
              "  https://www.nvidia.com/en-us/drivers/\n"
              "(This is checked before the engine and the model are downloaded.)",
    },
    "driver_unknown": {
        "ja": "ドライバーの版を読み取れませんでした。エンジンにはドライバー 580 以上が要ります。このまま続けます。\n"
              "後の点検で止まるときは、新しいドライバーを入れてください: https://www.nvidia.com/ja-jp/drivers/",
        "en": "The driver version could not be read. The engine needs driver 580 or newer. Going on.\n"
              "If the checks fail later, install a current driver: https://www.nvidia.com/en-us/drivers/",
    },
    "driver_skip": {
        "ja": "ドライバーが 580 より古いようですが（{d}）、--skip-driver-check の指定があるので続けます。エンジンは恐らく起動できません。",
        "en": "The driver looks older than 580 ({d}), but --skip-driver-check was given, so going on. The engine will most likely not start.",
    },
    "speed_only": {
        "ja": "GPU のメモリが 24 GB 未満なので、使えるのは最速モデルだけです（品質モデルは入りません）。",
        "en": "The GPU has less than 24 GB, so only the fastest model can be used (the quality model does not fit).",
    },
    "quality_no": {
        "ja": "品質モデルには GPU のメモリが 24 GB 以上要ります。--model speed で最速モデルを選んでください。",
        "en": "The quality model needs a GPU with 24 GB or more. Choose the fastest model with --model speed.",
    },
    "ctx_auto": {
        "ja": "文脈の長さは、起動のたびに GPU の空きメモリから決めます（いまの空きで {n} トークン。最大 262,144）。",
        "en": "The context length is chosen from the free GPU memory each time the model starts ({n} tokens with what is free now; "
              "at most 262,144).",
    },
    "ctx_head": {
        "ja": "文脈の場所を空けるため、補助モデルの出口の表（bdraft2_k8.shead）は読み込みません（1 回の計算が約 3% 長くなります）。",
        "en": "To make room for the context, the output head of the draft model (bdraft2_k8.shead) is not loaded "
              "(each pass takes about 3% longer).",
    },
    "ctx_short": {
        "ja": "いまの空きメモリでは足りません（GPU のメモリを {u} MiB 使用中）。GPU を使うほかのプログラムを閉じてから、もう一度試してください。",
        "en": "There is not enough free GPU memory now ({u} MiB in use). Close other programs that use the GPU and try again.",
    },
    "disk": {
        "ja": "空き容量: {f:.0f} GB（必要: 約 {n} GB）",
        "en": "Free disk space: {f:.0f} GB (needed: about {n} GB)",
    },
    "disk_short": {
        "ja": "空き容量が足りません。約 {n} GB を空けてから、もう一度試してください。",
        "en": "Not enough free disk space. Free about {n} GB and try again.",
    },
    "pick": {
        "ja": "どちらのモデルを使いますか？\n"
              "  1) 品質モデル〈おすすめ〉  元のモデルに近い。ファイル 12.4 GB\n"
              "  2) 最速モデル            いちばん速い。ファイル 10.3 GB\n"
              "番号を入れて Enter [{d}]: ",
        "en": "Which model?\n"
              "  1) quality model (recommended)  closest to the original. 12.4 GB file\n"
              "  2) fastest model                the fastest. 10.3 GB file\n"
              "Type the number and press Enter [{d}]: ",
    },
    "picked": {"ja": "モデル: {m}", "en": "Model: {m}"},
    "m_quality": {"ja": "品質モデル", "en": "quality model"},
    "m_speed": {"ja": "最速モデル", "en": "fastest model"},
    "checks_done": {"ja": "点検は済んでいます（飛ばします）。", "en": "The checks were done before (skipped)."},
    "engine_have": {"ja": "エンジンは準備済みです。", "en": "The engine is already in place."},
    "engine_local": {"ja": "このフォルダーにある {z} を使います。", "en": "Using {z} from this folder."},
    "engine_get": {"ja": "ダウンロード中: {u}", "en": "Downloading: {u}"},
    "sha_ok": {"ja": "SHA-256 は公開ページの値と同じです。", "en": "The SHA-256 matches the published one."},
    "sha_bad": {
        "ja": "SHA-256 が違います（ダウンロードが壊れたか、別のファイルです）。使わずに止めました。もう一度試してください。\n  期待: {e}\n  実際: {a}",
        "en": "The SHA-256 differs (broken download or a different file). It was not used; please try again.\n  expected: {e}\n  got:      {a}",
    },
    "dl_fail": {
        "ja": "ダウンロードできませんでした: {e}\nインターネットの接続を確かめて、もう一度 START.cmd をダブルクリックしてください（続きから始まります）。",
        "en": "The download failed: {e}\nCheck the internet connection and double-click START.cmd again (it continues where it stopped).",
    },
    "venv_have": {"ja": "Python の環境は準備済みです。", "en": "The Python environment is already in place."},
    "pip_fail": {
        "ja": "パッケージを入れられませんでした。上のメッセージを確かめてください。",
        "en": "Installing the package failed. See the messages above.",
    },
    "model_have": {"ja": "モデルは準備済みです: {p}", "en": "The model is already in place: {p}"},
    "model_get": {
        "ja": "Hugging Face から {r} をダウンロードします（約 {g} GB。回線によっては時間がかかります）。",
        "en": "Downloading {r} from Hugging Face (about {g} GB; this can take a while).",
    },
    "retry_dl": {"ja": "もう一度、別の方法で試します。", "en": "Trying once more with another transfer method."},
    "run": {"ja": "> yasakani {c}", "en": "> yasakani {c}"},
    "check_fail": {
        "ja": "「yasakani {c}」が失敗しました（終了コード {rc}）。上のメッセージを確かめてください。\n"
              "よくある原因: ほかのプログラムが GPU のメモリを使っている、ドライバーが古い、Visual C++ ランタイムが無い。\n"
              "FAQ: {r}/blob/main/docs/FAQ.ja.md",
        "en": "`yasakani {c}` failed (exit code {rc}). See the messages above.\n"
              "Common causes: another program is using the GPU memory, an old driver, or the Visual C++ runtime is missing.\n"
              "FAQ: {r}/blob/main/docs/FAQ.md",
    },
    "checks_ok": {
        "ja": "点検はすべて合格です。このパソコンは、公開されている基準と同じ出力を再現しました。",
        "en": "All checks passed. This computer reproduces the published reference output.",
    },
    "checks_other": {
        "ja": "info と verify は合格です。selfcheck の出力は RTX 5090 で作った基準と途中から違いましたが、基準の文章を読ませたときの予測は"
              "\nRTX 5090 と許容範囲内で一致し、この GPU 自身が生成した文章を読み直した結果も一致しました。"
              "\nこの GPU 向けの最適化と動作の確認はしていません。おかしな出力が続くときは、yasakani info の結果を添えて"
              "\nGitHub の Issues で知らせてください。このまま続けます。",
        "en": "info and verify passed. The output of selfcheck differs part-way from the reference, which was made on an RTX 5090, but when\n"
              "this GPU reads the reference text its predictions agree with the RTX 5090's within the tolerances, and it also agrees with\n"
              "itself when it reads its own output again. YASAKANI has not been optimized or tested on this GPU. If the output looks wrong,\n"
              "please report it on GitHub Issues with the output of yasakani info. Going on.",
    },
    "selfcheck_fail_other": {
        "ja": "selfcheck が不合格でした（終了コード {rc}）。この GPU の予測が、RTX 5090 の基準と許容範囲を超えて違うか、この GPU 自身が"
              "\n生成した文章と合いません（上の表示）。verify は合格なので、この GPU ではエンジンが正しく計算できていない恐れがあります。"
              "\nyasakani info と selfcheck の表示を添えて GitHub の Issues で知らせてください: {r}/issues"
              "\nそれでも使うときは、コマンドプロンプトで START.cmd --skip-checks と打つと点検を飛ばして進みます（出力が正しい保証はありません）。",
        "en": "selfcheck failed (exit code {rc}). This GPU's predictions differ from the RTX 5090's reference by more than the tolerances, or\n"
              "they do not agree with what this GPU generated itself (see above). verify passed, so the engine may not compute correctly\n"
              "on this GPU. Please report it on GitHub Issues with the output of yasakani info and of selfcheck: {r}/issues\n"
              "To use it anyway, type START.cmd --skip-checks in a command prompt: the checks are skipped (the output may be wrong).",
    },
    "menu": {
        "ja": "\n何をしますか？\n"
              "  1) 速さを測る             （yasakani bench）\n"
              "  2) 先読みあり・なしを比べる （技術の解説の数字を、このパソコンで確かめます。1 分ほど）\n"
              "  3) ブラウザーでチャット     （ブラウザーが開きます）\n"
              "  4) この画面でチャット       （yasakani chat。終わるときは /exit）\n"
              "  5) サーバーを起動           （OpenAI 互換。http://127.0.0.1:8765/v1 。止めるときは Ctrl+C）\n"
              "  6) 終わる\n"
              "番号を入れて Enter: ",
        "en": "\nWhat next?\n"
              "  1) measure the speed             (yasakani bench)\n"
              "  2) compare look-ahead on and off (checks the numbers of the technical notes on this PC; about a minute)\n"
              "  3) chat in the browser           (a browser window opens)\n"
              "  4) chat in this window           (yasakani chat; type /exit to leave)\n"
              "  5) start the server              (OpenAI-compatible, http://127.0.0.1:8765/v1 ; Ctrl+C stops it)\n"
              "  6) quit\n"
              "Type the number and press Enter: ",
    },
    "web_boot": {
        "ja": "サーバーを起動します（モデルの読み込みに 1 分ほどかかることがあります）。",
        "en": "Starting the server (loading the model can take about a minute).",
    },
    "web_ready": {
        "ja": "\nブラウザーでチャットできます: {u}\n"
              "（ブラウザーが自動で開きます。開かないときは、上のアドレスをブラウザーに入れてください）\n"
              "終わるときは、この画面で Enter を押してください。",
        "en": "\nChat in your browser: {u}\n"
              "(A browser window opens by itself. If it does not, type the address above into your browser.)\n"
              "Press Enter in this window to stop.",
    },
    "web_ready_manual": {
        "ja": "\nブラウザーでチャットできます: {u}\n"
              "（上のアドレスをブラウザーに入れてください）\n"
              "終わるときは、この画面で Enter を押してください。",
        "en": "\nChat in your browser: {u}\n"
              "(Type the address above into your browser.)\n"
              "Press Enter in this window to stop.",
    },
    "cmp_nofile": {
        "ja": "lookahead_check.py が見つかりません: {p}\nzip をもう一度展開してください。",
        "en": "lookahead_check.py was not found: {p}\nUnpack the zip again.",
    },
    "web_fail": {
        "ja": "サーバーを起動できませんでした。上のメッセージを確かめてください。",
        "en": "The server did not start. See the messages above.",
    },
    "web_nopage": {
        "ja": "chat.html が見つかりません: {p}\nzip をもう一度展開してください。",
        "en": "chat.html was not found: {p}\nUnpack the zip again.",
    },
    "web_end": {"ja": "チャットを終わりました。", "en": "The chat was stopped."},
    "load_note": {
        "ja": "（モデルを読み込みます。初回は 1 分ほどかかることがあります）",
        "en": "(loading the model; the first time this can take about a minute)",
    },
    "again": {
        "ja": "次からも START.cmd をダブルクリックするだけです（済んだ手順は飛ばします）。\n"
              "自分でコマンドを打つときの設定は {f} に書きました。",
        "en": "Next time just double-click START.cmd again (finished steps are skipped).\n"
              "The settings for typing commands yourself are in {f}.",
    },
    "bye": {"ja": "終了します。", "en": "Bye."},
    "stopped": {"ja": "中止しました。", "en": "Stopped."},
    "menu_update": {
        "ja": "  7) 新しい版を確かめる       （github.com に問い合わせます）\n",
        "en": "  7) check for a new version     (asks github.com)\n",
    },
    "upd_line": {
        "ja": "github.com に新しい版を問い合わせます（1 回だけ。このパソコンの情報は送りません。止めるときは --no-update-check）",
        "en": "Asking github.com for a new version (one request; nothing about this PC is sent; --no-update-check turns this off)",
    },
    "upd_latest": {"ja": "最新の版です（{v}）。", "en": "This is the latest version ({v})."},
    "upd_fail": {
        "ja": "新しい版を確かめられませんでした（{e}）。このまま続けます。",
        "en": "Could not check for a new version ({e}). Going on.",
    },
    "upd_found": {
        "ja": "新しい版があります: {o} → {n}\n  {u}",
        "en": "A new version is available: {o} -> {n}\n  {u}",
    },
    "upd_ask": {
        "ja": "いま更新しますか？（モデルはそのまま使います） [y/N]: ",
        "en": "Update now? (The models are kept.) [y/N]: ",
    },
    "upd_later": {
        "ja": "更新しません。あとで更新するときは、メニューの 7 か START.cmd --action update です。",
        "en": "Not updating. To update later, use 7 in the menu or START.cmd --action update.",
    },
    "upd_get": {"ja": "ダウンロード中: {u}", "en": "Downloading: {u}"},
    "upd_sha_ok": {"ja": "SHA-256 は、その版の SHA256SUMS.txt の値と同じです。", "en": "The SHA-256 matches the release's SHA256SUMS.txt."},
    "upd_bad": {
        "ja": "更新をやめました: {e}\n今の版はそのままです。",
        "en": "The update was stopped: {e}\nThe current version is unchanged.",
    },
    "upd_backup": {"ja": "今の起動ファイルを控えました: {p}", "en": "The current start files were copied to {p}"},
    "upd_done": {
        "ja": "起動ファイルを {n} に入れ替えました。新しい版を起動します（エンジンとパッケージは新しい版が入れ直します）。\n"
              "元に戻すときは: START.cmd --action rollback",
        "en": "The start files are now those of {n}. Starting the new version (it installs its own engine and package).\n"
              "To go back: START.cmd --action rollback",
    },
    "upd_web": {
        "ja": "チャットの画面から更新が頼まれました。サーバーを止めて更新します。",
        "en": "The chat page asked for the update. Stopping the server and updating.",
    },
    "upd_off": {
        "ja": "新しい版の自動の確認を止めました（--update-check で戻せます）。",
        "en": "The automatic check for a new version is off now (--update-check turns it on again).",
    },
    "upd_on": {"ja": "新しい版の自動の確認を使います（1 日に 1 回まで）。", "en": "The automatic check for a new version is on (at most once a day)."},
    "upd_dev": {
        "ja": "開発用: 更新の問い合わせ先を {u} にしています（GitHub ではありません）。",
        "en": "Development only: updates are looked up at {u} (not GitHub).",
    },
    "rb_none": {"ja": "戻せる控えがありません。", "en": "There is no backup to go back to."},
    "rb_ask": {
        "ja": "起動ファイルを {o} に戻しますか？ [y/N]: ",
        "en": "Put the start files of {o} back? [y/N]: ",
    },
    "rb_done": {
        "ja": "起動ファイルを {o} に戻しました。その版を起動します（エンジンとパッケージもその版のものに入れ直します）。",
        "en": "The start files of {o} are back. Starting that version (it installs its own engine and package again).",
    },
}


class Ui:
    def __init__(self, lang):
        self.lang = lang

    def t(self, key, **kw):
        s = T[key].get(self.lang) or T[key]["en"]
        return s.format(**kw) if kw else s

    def say(self, key, **kw):
        print(self.t(key, **kw), flush=True)

    def step(self, i, n, key):
        print("\n" + "=" * 72, flush=True)
        print(self.t("step", i=i, n=n, t=self.t(key)), flush=True)
        print("=" * 72, flush=True)


def ui_language():
    try:
        import ctypes
        lid = ctypes.windll.kernel32.GetUserDefaultUILanguage()
        return "ja" if (lid & 0x3FF) == 0x11 else "en"
    except Exception:
        return "en"


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def download(url, dst):
    tmp = dst + ".part"
    req = urllib.request.Request(url, headers={"User-Agent": "yasakani-quickstart/" + QS_VERSION})
    with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done, shown = 0, -1
        while True:
            b = r.read(1 << 18)
            if not b:
                break
            f.write(b)
            done += len(b)
            pct = int(done * 100 / total) if total else -1
            if pct != shown and (pct % 5 == 0 or pct < 0):
                shown = pct
                if total:
                    print("  %3d%%  %.1f / %.1f MB" % (pct, done / 1e6, total / 1e6), flush=True)
    os.replace(tmp, dst)


def safe_extract(zip_path, dst):
    root = os.path.abspath(dst)
    with zipfile.ZipFile(zip_path) as z:
        for m in z.infolist():
            p = os.path.abspath(os.path.join(root, m.filename))
            if not (p == root or p.startswith(root + os.sep)):
                raise RuntimeError("unsafe path in zip: " + m.filename)
        z.extractall(root)


def load_state(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_state(path, st):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(st, f, indent=1, sort_keys=True)
    os.replace(tmp, path)


def ask(prompt, default=""):
    try:
        s = input(prompt).strip()
    except EOFError:
        s = ""
    return s or default


def parse_gpus(text, fields=GPU_FIELDS):
    """Lines of ``nvidia-smi --query-gpu=... --format=csv,noheader,nounits`` -> dicts (index, uuid, name, cc, total, used,
    driver).  Values nvidia-smi could not give are None; lines without one value per field are skipped."""
    def val(s):
        s = s.strip()
        return None if (not s or s.startswith("[") or s.upper() == "N/A") else s

    def num(s):
        try:
            return int(round(float(s))) if s is not None else None
        except ValueError:
            return None
    out = []
    for n, line in enumerate(ln for ln in text.splitlines() if ln.strip()):
        parts = line.split(",")
        if len(parts) != len(fields):
            continue
        v = {k: val(p) for k, p in zip(fields, parts)}
        cc = None
        if v.get("compute_cap"):
            try:
                major, minor = v["compute_cap"].split(".")
                cc = (int(major), int(minor))
            except ValueError:
                cc = None
        index = num(v.get("index")) if "index" in v else n
        if not v.get("name") or index is None:
            continue
        out.append(dict(index=index, uuid=v.get("uuid"), name=v["name"], cc=cc, total=num(v.get("memory.total")),
                        used=num(v.get("memory.used")), driver=v.get("driver_version")))
    return out


def query_gpus(run=subprocess.run):
    """Every GPU nvidia-smi lists ([] for none).  Raises OSError when nvidia-smi cannot be run.  Drivers that do not know
    compute_cap are asked without it."""
    fallback = tuple(f for f in GPU_FIELDS if f != "compute_cap")
    for fields in (GPU_FIELDS, fallback):
        out = run(["nvidia-smi", "--query-gpu=" + ",".join(fields), "--format=csv,noheader,nounits"],
                  capture_output=True, text=True, timeout=60)
        if out.returncode == 0:
            return parse_gpus(out.stdout or "", fields)
    return []


def gpu_verdict(g):
    """"tested" (the RTX 5090), "other" (runs, not optimized or tested), "small" (under 16 GB), "old" (compute capability below 8.6)
    or "nocode" (9.x to 11.x, 13 and newer: the engine has no code for it)."""
    cc = g.get("cc")
    if cc is not None and cc < (8, 6):
        return "old"
    if cc is not None and not (cc < (9, 0) or cc[0] == 12):
        return "nocode"
    if re.search(r"(^|\s)RTX 5090$", g["name"].strip()) and cc in (None, (12, 0)):
        return "tested"
    if g.get("total") is not None and g["total"] < MIN_TOTAL_MIB:
        return "small"
    return "other"


def pick_gpu(gpus, environ=None):
    """The GPU to use: the one CUDA_VISIBLE_DEVICES names first, else the best usable one (the RTX 5090 first, then the one with the
    most memory).  -> (gpu, verdict, pin): ``pin`` is the GPU's UUID to put in CUDA_VISIBLE_DEVICES when the PC has more than one
    GPU, so that the engine runs on the GPU checked here (None: nothing to set)."""
    env = os.environ if environ is None else environ
    rank = {"tested": 0, "other": 1, "small": 2, "nocode": 3, "old": 4}
    if "CUDA_VISIBLE_DEVICES" in env:
        first = (env.get("CUDA_VISIBLE_DEVICES") or "").split(",")[0].strip()
        pci = env.get("CUDA_DEVICE_ORDER", "PCI_BUS_ID").strip().upper() == "PCI_BUS_ID"   # the package sets it when not set (0.1.5)
        named = [g for g in gpus if (first.upper().startswith("GPU-") and (g.get("uuid") or "").lower().startswith(first.lower()))
                 or (first.isdigit() and (len(gpus) == 1 or pci) and g["index"] == int(first))]
        if len(named) == 1:
            return named[0], gpu_verdict(named[0]), None
        worst = max(gpus, key=lambda g: (rank[gpu_verdict(g)], -(g.get("total") or 0)))   # not known which: the least able one
        return worst, gpu_verdict(worst), None
    best = min(gpus, key=lambda g: (rank[gpu_verdict(g)], -(g.get("total") or 0), -((g.get("total") or 0) - (g.get("used") or 0))))
    pin = best.get("uuid") if len(gpus) > 1 else None
    return best, gpu_verdict(best), pin


def estimate_context(free_mib, model):
    """The context the package will choose (yasakani/gpu.py, choose()): (tokens, head_off); tokens 0 = it does not fit."""
    w, d, h = MODEL_BYTES[model]
    mib = float(1 << 20)

    def ctx(files):
        room = free_mib - files / mib - OVERHEAD_MIB[model] - MARGIN_MIB
        n = max(0, int(room * mib // KV_BYTES_PER_TOKEN))
        return FULL_CONTEXT if n >= FULL_CONTEXT else (n // CONTEXT_STEP) * CONTEXT_STEP

    with_head = ctx(w + d + h)
    if with_head >= HEAD_OFF_BELOW:
        return with_head, False
    without = ctx(w + d)
    return (without, True) if without > with_head else (with_head, False)


def driver_major(text):
    """616 from nvidia-smi's driver_version "616.92" (580 from "580.65.06"); None when there is no usable value."""
    m = re.fullmatch(r"\s*(\d+)(?:\.\d+)*\s*", text or "")
    return int(m.group(1)) if m else None


def cuda_driver_version(load=None):
    """The newest CUDA version the installed driver supports, from cuDriverGetVersion() of nvcuda.dll (13040 = CUDA 13.4); None when it
    cannot be read.  This only asks the driver library for a number: nothing is started on the GPU.  ``load`` (for the tests) returns
    the library object."""
    try:
        import ctypes
        if load is None:
            lib = ctypes.WinDLL(os.path.join(os.environ.get("SystemRoot") or r"C:\Windows", "System32", "nvcuda.dll"))
        else:
            lib = load()
        v = ctypes.c_int(0)
        if lib.cuDriverGetVersion(ctypes.byref(v)) != 0 or v.value <= 0:
            return None
        return int(v.value)
    except Exception:  # noqa: BLE001 - no driver library, not Windows, ...: not known
        return None


def driver_label(driver, cuda, lang="en"):
    """What the messages show: the driver version and the newest CUDA version the driver supports (not the engine's), e.g.
    "616.92, supports up to CUDA 13.4" / "616.92、CUDA 13.4 まで対応", "616.92", "supports up to CUDA 12.8" or "?"."""
    parts = [driver] if driver else []
    if cuda:
        v = "%d.%d" % (cuda // 1000, (cuda % 1000) // 10)
        parts.append(("CUDA %s まで対応" if lang == "ja" else "supports up to CUDA %s") % v)
    return ("、" if lang == "ja" else ", ").join(parts) or "?"


def driver_verdict(driver, cuda=None):
    """"ok", "old" (older than the R580 branch, or a driver for CUDA 12 or older: the engine cannot start on it) or "unknown" (neither
    nvidia-smi's driver_version nor cuDriverGetVersion() gave a number).  Each number that is known must be new enough."""
    checks = []
    major = driver_major(driver)
    if major is not None:
        checks.append(major >= MIN_DRIVER)
    if cuda:
        checks.append(cuda >= MIN_CUDA_DRIVER_API)
    if not checks:
        return "unknown"
    return "ok" if all(checks) else "old"


# ------------------------------------------------------------------------------------------------------------------------ updates
class UpdateError(Exception):
    pass


class UpdateSource:
    """Where updates come from: GitHub's API for the latest release and this repository's download URLs.  ``dev_base`` (development
    only, --update-dev-base) points both at a server on this computer, e.g. http://127.0.0.1:8900, for testing an update end to end."""

    def __init__(self, dev_base=None):
        self.dev = bool(dev_base)
        if dev_base:
            m = re.fullmatch(r"http://(127\.0\.0\.1|localhost|\[::1\]):(\d{1,5})/?", dev_base)
            if not m:
                raise UpdateError("--update-dev-base must be http://127.0.0.1:<port> (a server on this computer)")
            base = "http://%s:%s/" % (m.group(1), m.group(2))
            self.api = base + "repos/yasakani-dev/yasakani/releases/latest"
            self.download = base + "yasakani-dev/yasakani/releases/download/"
            self.base = base.rstrip("/")
        else:
            self.api, self.download, self.base = UPDATE_API, UPDATE_DOWNLOAD, None


def parse_version(s):
    """(0, 1, 10) from "0.1.10" or "v0.1.10"; None for anything else (pre-release tags such as v0.1.7-test1 included)."""
    m = re.fullmatch(r"v?(\d{1,4})\.(\d{1,4})\.(\d{1,4})", str(s or "").strip())
    return tuple(int(x) for x in m.groups()) if m else None


def is_newer(new, old):
    a, b = parse_version(new), parse_version(old)
    return a is not None and b is not None and a > b


class _Redirects(urllib.request.HTTPRedirectHandler):
    """Release downloads redirect to GitHub's file host; follow https redirects only (http only to this computer in development)."""

    def __init__(self, dev):
        self.dev = dev

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not (newurl.startswith("https://") or (self.dev and re.match(r"http://(127\.0\.0\.1|localhost|\[::1\]):\d+/", newurl))):
            raise UpdateError("refused a redirect to " + newurl.split("?", 1)[0])
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def fetch_bytes(url, limit, timeout, dev=False):
    """GET ``url`` with a generic User-Agent and nothing else about this computer; at most ``limit`` bytes."""
    req = urllib.request.Request(url, headers={"User-Agent": UPDATE_UA, "Accept": "application/vnd.github+json, */*"})
    opener = urllib.request.build_opener(_Redirects(dev))
    with opener.open(req, timeout=timeout) as r:
        data = r.read(limit + 1)
    if len(data) > limit:
        raise UpdateError("the answer is larger than %d bytes" % limit)
    return data


def fetch_latest(src, timeout=5):
    """The latest release: {"version", "tag", "page", "assets"}; raises on any problem.  Drafts and pre-releases are refused."""
    try:
        j = json.loads(fetch_bytes(src.api, MAX_API, timeout, src.dev).decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        raise UpdateError("the answer of the release API is not JSON")
    if not isinstance(j, dict):
        raise UpdateError("the answer of the release API is not a release")
    tag = j.get("tag_name")
    if j.get("draft") or j.get("prerelease"):
        raise UpdateError("the latest release is a pre-release")
    v = parse_version(tag)
    if not (isinstance(tag, str) and tag.startswith("v") and v):
        raise UpdateError("the latest release has an unexpected tag")
    assets = {}
    for x in j.get("assets") or []:
        if isinstance(x, dict) and isinstance(x.get("name"), str) and isinstance(x.get("browser_download_url"), str):
            assets[x["name"]] = x["browser_download_url"]
    version = "%d.%d.%d" % v
    return {"version": version, "tag": tag, "page": REPO_URL + "/releases/tag/" + tag, "assets": assets}


def release_url(src, rel, name):
    """The download URL of ``name`` in that release.  Only URLs under this repository's download path of that tag are used."""
    base = src.download + rel["tag"] + "/"
    url = (rel.get("assets") or {}).get(name) or base + name
    if url != base + name:
        raise UpdateError("refused a download URL outside %s: %s" % (base, url))
    return url


def parse_sums(text):
    out = {}
    for line in text.splitlines():
        m = re.fullmatch(r"([0-9a-fA-F]{64}) [ *](\S+)", line.strip())
        if m:
            out[m.group(2)] = m.group(1).lower()
    return out


def check_start_zip(data, version):
    """The members of a quickstart zip -> {name: bytes}.  Refused: anything but plain files at the top whose names are START_FILES
    (no folders, no paths), duplicates, encrypted or too large members, a zip without quickstart.py and START.cmd, or a quickstart.py
    of another version."""
    if len(data) > MAX_START_ZIP:
        raise UpdateError("the zip is too large")
    try:
        z = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile:
        raise UpdateError("not a zip file")
    out = {}
    with z:
        infos = z.infolist()
        if len(infos) > len(START_FILES):
            raise UpdateError("the zip has too many members")
        for m in infos:
            if m.filename not in START_FILES or m.filename in out:
                raise UpdateError("unexpected member in the zip: %r" % m.filename)
            if m.is_dir() or m.flag_bits & 0x1 or m.file_size > MAX_START_FILE:
                raise UpdateError("member not allowed: %r" % m.filename)
            with z.open(m) as f:
                b = f.read(MAX_START_FILE + 1)
            if len(b) > MAX_START_FILE or len(b) != m.file_size:
                raise UpdateError("member not allowed: %r" % m.filename)
            out[m.filename] = b
    for need in ("quickstart.py", "START.cmd"):
        if need not in out:
            raise UpdateError("the zip has no " + need)
    if ('QS_VERSION = "%s"' % version).encode("ascii") not in out["quickstart.py"]:
        raise UpdateError("quickstart.py in the zip is not version " + version)
    return out


def download_update(ui, src, rel):
    """Downloads and checks the start files of ``rel`` -> {name: bytes}.  Nothing on disk changes."""
    zname = "yasakani-quickstart-%s.zip" % rel["version"]
    zurl, surl = release_url(src, rel, zname), release_url(src, rel, "SHA256SUMS.txt")
    ui.say("upd_get", u=surl)
    sums = parse_sums(fetch_bytes(surl, MAX_SUMS, 30, src.dev).decode("ascii", "replace"))
    if zname not in sums:
        raise UpdateError("SHA256SUMS.txt of %s does not list %s" % (rel["tag"], zname))
    ui.say("upd_get", u=zurl)
    data = fetch_bytes(zurl, MAX_START_ZIP, 60, src.dev)
    got = hashlib.sha256(data).hexdigest()
    if got != sums[zname]:
        raise UpdateError("the SHA-256 of %s differs from SHA256SUMS.txt (expected %s, got %s)" % (zname, sums[zname], got))
    ui.say("upd_sha_ok")
    return check_start_zip(data, rel["version"])


def _write_files(folder, files):
    """Writes every file as <name>.new first, then renames them over the old ones.  START.cmd may be the batch file that is running:
    START.cmd runs this script from inside one ( ... ) block, which cmd.exe has read completely before it runs it, so a new START.cmd
    does not change what the running one does after this script ends."""
    for n, b in files.items():
        with open(os.path.join(folder, n + ".new"), "wb") as f:
            f.write(b)
    for n in files:
        os.replace(os.path.join(folder, n + ".new"), os.path.join(folder, n))


def install_start_files(folder, files, old_version=None, before_replace=None):
    """Copies the current start files to backup/<old_version>/ and puts ``files`` in their place.  -> the backup folder.  If a file
    cannot be replaced, the files of the backup are put back.  ``before_replace`` is called once the backup is complete and before
    the first file is replaced (apply_update saves the pointer to the backup there, so --action rollback works after an
    interrupted update)."""
    old_version = old_version or QS_VERSION
    if not parse_version(old_version):
        raise UpdateError("bad version " + old_version)
    bdir = os.path.join(folder, "backup", old_version)
    tmp = bdir + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    kept = []
    for n in START_FILES:
        p = os.path.join(folder, n)
        if os.path.isfile(p):
            shutil.copy2(p, os.path.join(tmp, n))
            kept.append(n)
    with open(os.path.join(tmp, "backup.json"), "w", encoding="utf-8") as f:
        json.dump({"version": old_version, "files": kept, "made": int(time.time())}, f, indent=1)
    if os.path.isdir(bdir):
        shutil.rmtree(bdir)
    os.replace(tmp, bdir)
    if before_replace is not None:
        before_replace(bdir)
    try:
        _write_files(folder, files)
    except OSError as e:
        restore = {}
        for n in kept:
            with open(os.path.join(bdir, n), "rb") as f:
                restore[n] = f.read()
        _write_files(folder, restore)
        e.restored = True                               # the old start files are back in place
        raise
    return bdir


def read_backup(folder, version):
    """The files of backup/<version>/ -> {name: bytes}; raises when there is no usable backup."""
    if not parse_version(version):
        raise UpdateError("bad version")
    bdir = os.path.join(folder, "backup", version)
    try:
        with open(os.path.join(bdir, "backup.json"), encoding="utf-8") as f:
            man = json.load(f)
    except (OSError, ValueError):
        raise UpdateError("no backup of " + version)
    out = {}
    for n in man.get("files") or []:
        if n not in START_FILES:
            raise UpdateError("unexpected file in the backup: %r" % n)
        p = os.path.join(bdir, n)
        if os.path.getsize(p) > MAX_START_FILE:
            raise UpdateError("file too large in the backup: %r" % n)
        with open(p, "rb") as f:
            out[n] = f.read()
    if "quickstart.py" not in out or "START.cmd" not in out:
        raise UpdateError("the backup of %s is not complete" % version)
    return out


def check_disabled(a, st, environ=None):
    env = os.environ if environ is None else environ
    return bool(getattr(a, "no_update_check", False)) or env.get("YASAKANI_NO_UPDATE_CHECK", "") not in ("", "0") \
        or st.get("update_check") is False


def update_check(ui, st, save, src, force=False, now=None, disabled=False):
    """The check of the menu (at most once a day) or, with ``force``, of the menu item and --action update.  -> the newer release
    or None.  Never raises: a failed check prints one line and goes on."""
    now = time.time() if now is None else now
    if disabled and not force:
        return None
    last = st.get("update_checked") or 0
    if not force and 0 <= now - last < CHECK_EVERY:
        c = st.get("update_latest")
        if isinstance(c, dict) and is_newer(c.get("version"), QS_VERSION) and c.get("version") != st.get("update_skip") \
                and parse_version(c.get("tag")):
            return {"version": c["version"], "tag": c["tag"], "page": REPO_URL + "/releases/tag/" + c["tag"], "assets": {}}
        return None
    print(ui.t("upd_line"), flush=True)
    st["update_checked"] = now
    try:
        rel = fetch_latest(src)
    except Exception as e:  # noqa: BLE001 - offline, timeout, refused, rate limit, ...: never stop the launcher for it
        if hasattr(e, "close"):
            e.close()                                   # an HTTPError holds the answer
        save()
        ui.say("upd_fail", e=str(e) if isinstance(e, UpdateError) else type(e).__name__)
        return None
    st["update_latest"] = {"version": rel["version"], "tag": rel["tag"]}
    save()
    if not is_newer(rel["version"], QS_VERSION) or (not force and rel["version"] == st.get("update_skip")):
        if force or not is_newer(rel["version"], QS_VERSION):
            ui.say("upd_latest", v=QS_VERSION)
        return None
    return rel


def run_quickstart(folder, args):
    """Runs the quickstart.py of ``folder`` (the new one after an update) with the system Python that runs this one."""
    print(flush=True)
    return subprocess.call([sys.executable, os.path.join(folder, "quickstart.py")] + list(args))


def apply_update(ui, folder, rel, src, yes, after_args, st, save):
    """Asks (unless ``yes``), downloads, checks, backs up, replaces, then runs the new quickstart.py.  -> its exit code, 9 when the
    update failed (nothing changed), None when the answer was no."""
    ui.say("upd_found", o=QS_VERSION, n=rel["version"], u=rel["page"])
    if not yes and ask(ui.t("upd_ask"), "n").lower() not in ("y", "yes"):
        ui.say("upd_later")
        return None
    had, before = "update_backup" in st, st.get("update_backup")

    def keep_pointer(bdir):
        # saved before the first start file is replaced: if the update stops halfway, --action rollback finds the backup
        st["update_backup"] = {"version": QS_VERSION, "from": rel["version"]}
        save()

    try:
        files = download_update(ui, src, rel)
        bdir = install_start_files(folder, files, before_replace=keep_pointer)
    except Exception as e:  # noqa: BLE001 - network, checksum, zip, disk: the current version stays
        if st.get("update_backup") != before and getattr(e, "restored", False):   # the old files are back: keep the earlier pointer
            if had:
                st["update_backup"] = before
            else:
                st.pop("update_backup", None)
            save()
        ui.say("upd_bad", e=str(e) if isinstance(e, UpdateError) else "%s: %s" % (type(e).__name__, e))
        return 9
    st.pop("update_skip", None)
    save()
    ui.say("upd_backup", p=bdir)
    ui.say("upd_done", n=rel["version"])
    return run_quickstart(folder, after_args)


def update_answered(a):
    """--yes answers the update question only together with --action update.  The daily check at the menu and the menu item
    always ask, also when the launcher runs with --yes (for example after an update, which passes --yes on)."""
    return bool(getattr(a, "yes", False)) and getattr(a, "action", None) == "update"


def apply_rollback(ui, folder, st, save, yes, after_args):
    b = st.get("update_backup") or {}
    try:
        files = read_backup(folder, b.get("version", ""))
    except UpdateError:
        ui.say("rb_none")
        return 1
    if not yes and ask(ui.t("rb_ask", o=b["version"]), "n").lower() not in ("y", "yes"):
        ui.say("stopped")
        return 1
    _write_files(folder, files)
    st.pop("update_backup", None)
    if b.get("from"):
        st["update_skip"] = b["from"]                 # the daily check does not offer that version again (the menu item still does)
    save()
    ui.say("rb_done", o=b["version"])
    return run_quickstart(folder, after_args)


# ---------------------------------------------------------------------------------------------------------------- browser chat
def port_is_free(port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            s.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        s.bind(("127.0.0.1", port))
        return True
    except OSError:
        return False
    finally:
        s.close()


def pick_port(preferred, avoid=()):
    for p in [preferred] + list(range(preferred + 10, preferred + 60)):
        if p not in avoid and port_is_free(p):
            return p
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]
    finally:
        s.close()


class LocalServer(ThreadingHTTPServer):
    """Listens on 127.0.0.1 only and never shares its port."""
    allow_reuse_address = False
    daemon_threads = True

    def server_bind(self):
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


class UpdateRequest:
    """The update button of the chat page.  It exists only when this launcher found a newer version.  The page gets the
    one-time ``token`` from /webchat/config (which only pages of this address can read) and sends it in the X-Yasakani-Token header
    of a POST to /webchat/update; the request must also name this address in Host and Origin.  The token works once."""

    HEADER = "X-Yasakani-Token"

    def __init__(self, rel):
        self.rel = rel
        self.token = hashlib.sha256(os.urandom(32)).hexdigest()
        self.event = threading.Event()
        self._lock = threading.Lock()

    def take(self, token):
        with self._lock:
            if not self.token or not token or not hmac.compare_digest(self.token.encode(), token.encode("utf-8", "replace")):
                return False
            self.token = None
            return True


def make_web_handler(page, conf, engine_port, web_port, update=None):
    """The handler of the browser chat: the page, its settings, and three paths passed on to `yasakani serve`.

    Like `yasakani serve`, it answers only requests that name this address in ``Host`` and, if they carry an ``Origin``, come from
    this address - so a web page of another site that is open in the browser cannot talk to the model through it.  With ``update``
    (an UpdateRequest) it also answers POST /webchat/update; without it that path does not exist."""
    own = ("127.0.0.1:%d" % web_port, "localhost:%d" % web_port)
    origins = tuple("http://" + h for h in own)
    if update is not None:
        conf = dict(conf, update={"version": update.rel["version"], "page": update.rel["page"]}, update_token=update.token)
    conf_bytes = json.dumps(conf).encode("utf-8")

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        timeout = 300
        server_version = "yasakani-quickstart/" + QS_VERSION
        sys_version = ""

        def log_message(self, *args):
            pass

        def _refused(self):
            if (self.headers.get("Host") or "").lower() not in own:
                return "host"
            origin = self.headers.get("Origin")
            if origin is not None and origin.lower() not in origins:
                return "origin"
            if (self.headers.get("Sec-Fetch-Site") or "none").lower() not in ("same-origin", "none"):
                return "site"
            return None

        def _send(self, code, ctype, body, extra=(), close=False):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cross-Origin-Resource-Policy", "same-origin")
            self.send_header("Referrer-Policy", "no-referrer")
            for k, v in extra:
                self.send_header(k, v)
            if close:
                self.send_header("Connection", "close")
                self.close_connection = True
            self.end_headers()
            self.wfile.write(body)

        def _error(self, code, message, kind, close=False):
            self._send(code, "application/json; charset=utf-8",
                       json.dumps({"error": {"message": message, "type": kind}}).encode("utf-8"), close=close)

        def _relay(self, method, path, body=None):
            conn = http.client.HTTPConnection("127.0.0.1", engine_port, timeout=900)
            started = False
            try:
                headers = {"Accept": "*/*"}
                if body is not None:
                    headers["Content-Type"] = "application/json"
                conn.request(method, path, body=body, headers=headers)
                resp = conn.getresponse()
                ctype = resp.getheader("Content-Type") or "application/octet-stream"
                if ctype.lower().startswith("text/event-stream"):
                    started = True
                    self.send_response(resp.status)
                    self.send_header("Content-Type", ctype)
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("X-Content-Type-Options", "nosniff")
                    self.send_header("Connection", "close")
                    self.end_headers()
                    self.close_connection = True
                    while True:
                        chunk = resp.read1(65536)
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        self.wfile.flush()
                else:
                    data = resp.read()
                    started = True
                    self._send(resp.status, ctype, data)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                self.close_connection = True      # the browser went away; closing the connection below stops the generation
            except (OSError, http.client.HTTPException) as ex:
                self.close_connection = True
                if not started:
                    try:
                        self._error(502, "the YASAKANI server did not answer (%s)" % type(ex).__name__, "upstream", close=True)
                    except OSError:
                        pass
            finally:
                conn.close()

        def do_GET(self):
            why = self._refused()
            if why:
                return self._error(403, "forbidden: " + why, "forbidden", close=True)
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index.html"):
                return self._send(200, "text/html; charset=utf-8", page, (("Content-Security-Policy", PAGE_CSP),))
            if path == "/webchat/config":
                return self._send(200, "application/json; charset=utf-8", conf_bytes)
            if path in ("/health", "/v1/models"):
                return self._relay("GET", path)
            if path == "/favicon.ico":
                return self._send(404, "text/plain; charset=utf-8", b"")
            return self._error(404, "not found", "not_found")

        def do_POST(self):
            why = self._refused()
            if why:
                return self._error(403, "forbidden: " + why, "forbidden", close=True)
            if update is not None and self.path.split("?", 1)[0] == "/webchat/update":
                return self._update()
            if self.path.split("?", 1)[0] != "/v1/chat/completions":
                return self._error(404, "not found", "not_found", close=True)
            try:
                n = int(self.headers.get("Content-Length", "") or -1)
            except ValueError:
                n = -1
            if n < 0 or n > MAX_BODY:
                return self._error(413, "request body too large or length missing", "invalid_request_error", close=True)
            self._relay("POST", "/v1/chat/completions", self.rfile.read(n))

        def _update(self):
            if (self.headers.get("Origin") or "").lower() not in origins:      # browsers send Origin with every POST of fetch()
                return self._error(403, "forbidden: origin", "forbidden", close=True)
            try:
                n = int(self.headers.get("Content-Length", "") or 0)
            except ValueError:
                n = -1
            if n < 0 or n > 4096:
                return self._error(413, "request body too large", "invalid_request_error", close=True)
            if n:
                self.rfile.read(n)
            if not update.take(self.headers.get(UpdateRequest.HEADER)):
                return self._error(403, "forbidden: token", "forbidden", close=True)
            self._send(202, "application/json; charset=utf-8",
                       json.dumps({"status": "updating", "version": update.rel["version"]}).encode("utf-8"), close=True)
            self.wfile.flush()
            update.event.set()

    return Handler


def get_json(port, path, headers=None, timeout=5):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        conn.request("GET", path, headers=headers or {})
        r = conn.getresponse()
        data = r.read()
        try:
            return r.status, json.loads(data or b"{}"), r
        except ValueError:
            return r.status, {}, r
    finally:
        conn.close()


def wait_ready(port, proc, seconds=900):
    t0 = time.time()
    while time.time() - t0 < seconds:
        if proc.poll() is not None:
            return False
        try:
            code, body, _ = get_json(port, "/health")
            if code == 200 and body.get("status") == "ok":
                return True
        except (OSError, http.client.HTTPException):
            pass
        time.sleep(1.0)
    return False


def web_selftest(web_port):
    """Checks the browser chat once, through the same address the browser uses.  Prints one line per check."""
    results = []

    def line(name, good, extra=""):
        results.append(bool(good))
        print("  %-46s %s%s" % (name, "PASS" if good else "FAIL", ("   " + extra) if extra else ""), flush=True)

    def raw(method, path, body=None, headers=None):
        conn = http.client.HTTPConnection("127.0.0.1", web_port, timeout=600)
        try:
            h = dict(headers or {})
            if body is not None:
                h.setdefault("Content-Type", "application/json")
            conn.request(method, path, body=body, headers=h)
            r = conn.getresponse()
            return r.status, r.getheader("Content-Type") or "", r.getheader("Content-Security-Policy") or "", r.read()
        finally:
            conn.close()

    print("\nbrowser chat self-test (http://127.0.0.1:%d/)" % web_port, flush=True)
    try:
        code, ctype, csp, data = raw("GET", "/")
        line("the page is served", code == 200 and ctype.startswith("text/html") and b"YASAKANI" in data and "default-src 'none'" in csp,
             "%d bytes" % len(data))
        code, ctype, _, data = raw("GET", "/webchat/config")
        line("the settings of the page", code == 200 and json.loads(data).get("version") == QS_VERSION)
        code, ctype, _, data = raw("GET", "/health")
        h = json.loads(data) if code == 200 else {}
        line("the model is ready", h.get("status") == "ok", "context %s tokens" % h.get("max_context_tokens"))
        code, _, _, _ = raw("GET", "/health", headers={"Origin": "http://example.com"})
        line("a page of another site is refused", code == 403, "HTTP %d" % code)
        code, _, _, _ = raw("GET", "/health", headers={"Host": "example.com"})
        line("another host name is refused", code == 403, "HTTP %d" % code)
        code, _, _, _ = raw("POST", "/v1/chat/completions", body=b"{}", headers={"Sec-Fetch-Site": "cross-site"})
        line("a cross-site request is refused", code == 403, "HTTP %d" % code)
        code, _, _, _ = raw("GET", "/yasakani_env.cmd")
        line("other files are not served", code == 404, "HTTP %d" % code)
        body = json.dumps({"messages": [{"role": "user", "content": "Write the numbers from 1 to 120, separated by spaces."}],
                           "stream": True, "stream_options": {"include_usage": True}, "temperature": 0, "max_tokens": 480,
                           "chat_template_kwargs": {"enable_thinking": False}}).encode("utf-8")
        conn = http.client.HTTPConnection("127.0.0.1", web_port, timeout=600)
        try:
            conn.request("POST", "/v1/chat/completions", body=body, headers={"Content-Type": "application/json"})
            r = conn.getresponse()
            n, text, usage, t_first, t_last, buf = 0, [], None, 0.0, 0.0, b""
            while True:
                chunk = r.read1(65536)
                if not chunk:
                    break
                buf += chunk
                while b"\n\n" in buf:
                    ev, buf = buf.split(b"\n\n", 1)
                    if not ev.startswith(b"data:"):
                        continue
                    d = ev[5:].strip()
                    if d == b"[DONE]":
                        continue
                    o = json.loads(d)
                    if o.get("usage"):
                        usage = o["usage"]
                    for c in o.get("choices") or []:
                        piece = (c.get("delta") or {}).get("content")
                        if piece:
                            n += 1
                            text.append(piece)
                            t_last = time.perf_counter()
                            t_first = t_first or t_last
            secs = t_last - t_first
            got = usage["completion_tokens"] if usage else n
            rate = (got - 1) / secs if got > 16 and secs > 0.1 else 0.0
            answer = "".join(text)
            line("an answer arrives as a stream", r.status == 200 and n > 16 and " 20 " in answer,
                 "%d tokens, %.0f tokens/s" % (got, rate))
        finally:
            conn.close()
    except Exception as ex:  # noqa: BLE001 - any failure is a failed self-test
        line("self-test ran to the end", False, "%s: %s" % (type(ex).__name__, ex))
    good = all(results)
    print("browser chat self-test: %s" % ("PASS" if good else "FAIL"), flush=True)
    return good


def wait_for_stop(proc, update, poll=0.2):
    """Waits until Enter is pressed in this window, the server ends, or the page asked for the update (-> "update").  Enter is read
    with msvcrt, not input(): a thread left waiting in input() would take the first line typed into the next quickstart.py."""
    try:
        import msvcrt
    except ImportError:
        msvcrt = None
    tty = sys.stdin is not None and sys.stdin.isatty()
    while True:
        if update is not None and update.event.is_set():
            return "update"
        if proc.poll() is not None:
            return "ended"
        if tty and msvcrt is not None:
            while msvcrt.kbhit():
                if msvcrt.getwch() in ("\r", "\n"):
                    return "enter"
        time.sleep(poll)


def run_web(ui, a, vpy, env, mdir, dll, model, update_rel=None):
    page_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chat.html")
    if not os.path.isfile(page_path):
        ui.say("web_nopage", p=page_path)
        return 7
    with open(page_path, "rb") as f:
        page = f.read()
    eport = pick_port(ENGINE_PORT)
    wport = pick_port(getattr(a, "web_port", None) or WEB_PORT, avoid=(eport,))
    a.web_port_used = wport
    update = UpdateRequest(update_rel) if update_rel else None
    ui.say("web_boot")
    proc = subprocess.Popen([vpy, "-m", "yasakani", "serve", "--model-dir", mdir, "--dll", dll, "--port", str(eport)], env=env)
    httpd = None
    try:
        if not wait_ready(eport, proc):
            ui.say("web_fail")
            return 7
        conf = {"lang": ui.lang, "model": "custom" if a.model_dir else model, "version": QS_VERSION, "repo": REPO_URL, "support": SUPPORT_URL}
        httpd = LocalServer(("127.0.0.1", wport), make_web_handler(page, conf, eport, wport, update))
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        url = "http://127.0.0.1:%d/" % wport
        if a.web_test:
            return 0 if web_selftest(wport) else 8
        ui.say("web_ready_manual" if a.no_browser else "web_ready", u=url)
        if not a.no_browser:
            try:
                webbrowser.open(url)
            except Exception:  # noqa: BLE001 - the address is printed above
                pass
        try:
            why = wait_for_stop(proc, update)
        except KeyboardInterrupt:
            why = "enter"
        if why == "update":
            ui.say("upd_web")
            return UPDATE_RC
        return 0
    finally:
        if httpd is not None:
            httpd.shutdown()
            httpd.server_close()
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
        ui.say("web_end")


def main(argv=None):
    ap = argparse.ArgumentParser(description="YASAKANI quickstart")
    ap.add_argument("--model", choices=sorted(MODELS), help="skip the question: quality or speed")
    ap.add_argument("--model-dir", help="use this model folder instead of downloading one")
    ap.add_argument("--engine-zip", help="use this engine zip instead of downloading it (its SHA-256 is still checked)")
    ap.add_argument("--home", help="where to keep everything (default: the folder of this file)")
    ap.add_argument("--action", choices=("menu", "bench", "compare", "web", "chat", "serve", "none", "update", "rollback"),
                    default="menu",
                    help="what to do after the checks (default: show the menu); compare = look-ahead on and off, "
                         "web = chat in the browser; update = look for a new version now and install it after a yes "
                         "(with --yes without asking); rollback = put back the start files of the version before the last update")
    ap.add_argument("--no-update-check", action="store_true",
                    help="turn the automatic check for a new version off (kept in quickstart_state.json)")
    ap.add_argument("--update-check", action="store_true", help="turn the automatic check for a new version on again")
    ap.add_argument("--update-dev-base", metavar="URL",
                    help="development only: look for updates at http://127.0.0.1:<port> (a test server on this computer) "
                         "instead of GitHub")
    ap.add_argument("--web-port", type=int, help="with the browser chat: the port to try first (an update keeps the page's port)")
    ap.add_argument("--no-browser", action="store_true", help="with the browser chat: do not open a browser window")
    ap.add_argument("--web-test", action="store_true", help="start the browser chat, check it once and stop")
    ap.add_argument("--lang", choices=("ja", "en"), help="language of the messages (default: the Windows display language)")
    ap.add_argument("--yes", action="store_true", help="answer yes to questions (the update question only with --action update)")
    ap.add_argument("--skip-checks", action="store_true", help="do not run info / verify / selfcheck")
    ap.add_argument("--skip-driver-check", action="store_true",
                    help="do not stop when the NVIDIA driver is older than 580 (the engine will most likely not start)")
    a = ap.parse_args(argv)
    if a.web_test:
        a.action = "web"

    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(errors="replace")
        except Exception:
            pass
    ui = Ui(a.lang or ui_language())
    print()
    ui.say("banner", v=QS_VERSION)
    print(REPO_URL)

    if sys.platform != "win32":
        ui.say("not_windows")
        return 2
    if sys.version_info < (3, 10):
        ui.say("old_python", v=sys.version.split()[0])
        return 2
    home = os.path.abspath(a.home or os.path.dirname(os.path.abspath(__file__)))
    try:
        home.encode("ascii")
    except UnicodeEncodeError:
        ui.say("bad_path", p=home)
        return 2
    os.makedirs(home, exist_ok=True)
    state_path = os.path.join(home, "quickstart_state.json")
    st = load_state(state_path)
    n = 7

    def save():
        save_state(state_path, st)

    # updates: the folder of the start files, where they come from, and the arguments the next quickstart.py gets
    qs_dir = os.path.dirname(os.path.abspath(__file__))
    st["quickstart_dir"] = qs_dir                       # `yasakani update` finds the launcher through this
    if a.no_update_check or a.update_check:
        st["update_check"] = not a.no_update_check
        ui.say("upd_off" if a.no_update_check else "upd_on")
    save()
    try:
        src = UpdateSource(a.update_dev_base)
    except UpdateError as e:
        print(e)
        return 2
    if src.dev:
        ui.say("upd_dev", u=src.base)

    def after_args(action, web_port=None):
        out = ["--action", action, "--lang", ui.lang]
        for flag, val in (("--home", a.home), ("--model-dir", a.model_dir), ("--update-dev-base", a.update_dev_base)):
            if val:
                out += [flag, val]
        for flag, on in (("--skip-checks", a.skip_checks), ("--skip-driver-check", a.skip_driver_check), ("--yes", a.yes),
                         ("--no-browser", a.no_browser or action == "web")):
            if on:
                out.append(flag)
        if web_port:
            out += ["--web-port", str(web_port)]
        return out

    if a.action == "update":                            # before the checks of this computer: they are the new version's job
        rel = update_check(ui, st, save, src, force=True)
        if not rel:
            return 0
        rc = apply_update(ui, qs_dir, rel, src, update_answered(a), after_args("none"), st, save)
        return 0 if rc is None else rc
    if a.action == "rollback":
        return apply_rollback(ui, qs_dir, st, save, a.yes, after_args("none"))

    # 1. this computer
    ui.step(1, n, "s_check")
    print("Python %s" % sys.version.split()[0])
    try:
        gpus = query_gpus()
    except (OSError, subprocess.SubprocessError):
        ui.say("no_smi")
        return 2
    if not gpus:
        ui.say("no_gpu")
        return 2
    for g in gpus:
        ui.say("gpu_is", g=g["name"], c="%d.%d" % g["cc"] if g["cc"] else "?", t=g["total"] if g["total"] is not None else "?")
    gpu, verdict, pin = pick_gpu(gpus)
    if len(gpus) > 1:
        ui.say("gpu_pick", n=len(gpus), g=gpu["name"])
    if verdict == "old":
        ui.say("gpu_old", c="%d.%d" % gpu["cc"])
        return 2
    if verdict == "nocode":
        ui.say("gpu_nocode", c="%d.%d" % gpu["cc"])
        return 2
    if verdict == "small":
        ui.say("gpu_small", t=gpu["total"])
        return 2
    ui.say("gpu_tested" if verdict == "tested" else "gpu_other")
    cuda = cuda_driver_version()
    dv, dlabel = driver_verdict(gpu["driver"], cuda), driver_label(gpu["driver"], cuda, ui.lang)
    if dv == "old":                                     # before anything is downloaded: the engine cannot start on this driver
        if not a.skip_driver_check:
            ui.say("driver_old", d=dlabel)
            return 2
        ui.say("driver_skip", d=dlabel)
    elif dv == "unknown":
        ui.say("driver_unknown")
    else:
        ui.say("driver_ok", d=dlabel)
    quality_fits = gpu["total"] is None or gpu["total"] >= QUALITY_MIN_TOTAL_MIB

    # 2. model
    ui.step(2, n, "s_model")
    ok = st.setdefault("models_ok", {})
    model = a.model
    if not quality_fits and not a.model_dir:
        if model == "quality":
            ui.say("quality_no")
            return 2
        model = "speed"
        ui.say("speed_only")
    if not model:
        prev = st.get("model") or "quality"
        if a.yes or a.model_dir or a.action != "menu":
            model = prev
        else:
            d = "2" if prev == "speed" else "1"
            model = "speed" if ask(ui.t("pick", d=d), d) == "2" else "quality"
    ui.say("picked", m=ui.t("m_" + model))
    if gpu["total"] is not None and gpu["used"] is not None and gpu["total"] < FULL_CONTEXT_MIN_TOTAL_MIB and not a.model_dir:
        n_ctx, head_off = estimate_context(gpu["total"] - gpu["used"], model)
        if n_ctx < MIN_CONTEXT:
            ui.say("ctx_short", u=gpu["used"])
        else:
            ui.say("ctx_auto", n="{:,}".format(n_ctx))
            if head_off:
                ui.say("ctx_head")
    mdir = os.path.abspath(a.model_dir) if a.model_dir else os.path.join(home, "models", model)
    try:
        mdir.encode("ascii")
    except UnicodeEncodeError:
        ui.say("bad_path", p=mdir)
        return 2
    have_model = os.path.isfile(os.path.join(mdir, "yasakani.json"))
    need = 2 if (a.model_dir or have_model) else MODELS[model]["need_gb"]
    free = shutil.disk_usage(home).free / 1e9
    ui.say("disk", f=free, n=need)
    if free < need:
        ui.say("disk_short", n=need)
        return 2
    st["model"] = model
    save_state(state_path, st)

    # 3. engine
    ui.step(3, n, "s_engine")
    edir = os.path.join(home, "engine")
    dll = os.path.join(edir, "yskn.dll")
    if st.get("engine_sha") == ENGINE_SHA256 and os.path.isfile(dll) and os.path.isfile(os.path.join(edir, "pyproject.toml")):
        ui.say("engine_have")
    else:
        local = a.engine_zip or os.path.join(home, ENGINE_ZIP)
        if os.path.isfile(local):
            ui.say("engine_local", z=os.path.basename(local))
            zpath, fetched = local, False
        else:
            os.makedirs(os.path.join(home, "downloads"), exist_ok=True)
            zpath, fetched = os.path.join(home, "downloads", ENGINE_ZIP), True
            ui.say("engine_get", u=ENGINE_URL)
            try:
                download(ENGINE_URL, zpath)
            except Exception as e:  # network errors of any kind
                ui.say("dl_fail", e="%s: %s" % (type(e).__name__, e))
                return 3
        got = sha256_of(zpath)
        if got != ENGINE_SHA256:
            if fetched:
                os.remove(zpath)
            ui.say("sha_bad", e=ENGINE_SHA256, a=got)
            return 3
        ui.say("sha_ok")
        if os.path.isdir(edir):
            shutil.rmtree(edir)
        safe_extract(zpath, edir)
        st["engine_sha"], st["venv_ok"] = ENGINE_SHA256, False
        save_state(state_path, st)

    # 4. environment
    ui.step(4, n, "s_venv")
    vdir = os.path.join(home, "venv")
    vpy = os.path.join(vdir, "Scripts", "python.exe")
    if st.get("venv_ok") and os.path.isfile(vpy):
        ui.say("venv_have")
    else:
        if not os.path.isfile(vpy):
            subprocess.check_call([sys.executable, "-m", "venv", vdir])
        rc = subprocess.call([vpy, "-m", "pip", "install", "--disable-pip-version-check", edir, "huggingface_hub"])
        if rc != 0:
            ui.say("pip_fail")
            return 4
        st["venv_ok"] = True
        save_state(state_path, st)

    env = dict(os.environ)
    env["YASAKANI_DLL"] = dll
    env["YASAKANI_MODEL_DIR"] = mdir
    if pin:
        env["CUDA_VISIBLE_DEVICES"] = pin            # more than one GPU: the engine runs on the GPU checked in step 1

    # 5. model files
    ui.step(5, n, "s_dl")
    got = st.setdefault("models_downloaded", {})
    if a.model_dir or (got.get(model) and have_model):
        ui.say("model_have", p=mdir)
    else:
        repo = MODELS[model]["repo"]
        ui.say("model_get", r=repo, g=MODELS[model]["need_gb"] - 1)
        os.makedirs(mdir, exist_ok=True)
        code = ("import sys\nfrom huggingface_hub import snapshot_download\n"
                "snapshot_download(repo_id=sys.argv[1], local_dir=sys.argv[2])\n")
        rc = subprocess.call([vpy, "-c", code, repo, mdir], env=env)
        if rc != 0:
            ui.say("retry_dl")
            env2 = dict(env)
            env2["HF_HUB_DISABLE_XET"] = "1"
            rc = subprocess.call([vpy, "-c", code, repo, mdir], env=env2)
        if rc != 0:
            ui.say("dl_fail", e="exit code %d" % rc)
            return 5
        got[model] = True
        save_state(state_path, st)

    def yk(cmd, *extra):
        args = [vpy, "-m", "yasakani", cmd, "--model-dir", mdir]
        if cmd != "verify":
            args += ["--dll", dll]
        args += list(extra)
        print()
        ui.say("run", c=cmd)
        return subprocess.call(args, env=env)

    # 6. checks
    ui.step(6, n, "s_verify")
    if a.skip_checks or (ok.get(model) and not a.model_dir):
        ui.say("checks_done")
    else:
        differs = False
        for cmd in ("info", "verify", "selfcheck"):
            if cmd == "selfcheck":
                ui.say("load_note")
            rc = yk(cmd)
            if rc == 3 and cmd == "selfcheck" and verdict != "tested":
                differs = True
            elif rc == 1 and cmd == "selfcheck" and verdict != "tested":   # 0.1.4: the predictions or the self-consistency failed
                ui.say("selfcheck_fail_other", rc=rc, r=REPO_URL)
                return 6
            elif rc != 0:
                ui.say("check_fail", c=cmd, rc=rc, r=REPO_URL)
                return 6
        print()
        ui.say("checks_other" if differs else "checks_ok")
        if not a.model_dir:
            ok[model] = True
            save_state(state_path, st)

    envfile = os.path.join(home, "yasakani_env.cmd")
    with open(envfile, "w", encoding="ascii", newline="\r\n") as f:
        f.write("@echo off\nrem Run this in a command prompt, then type: yasakani chat   (or serve, bench, info ...)\n")
        f.write('set "YASAKANI_DLL=%s"\nset "YASAKANI_MODEL_DIR=%s"\nset "PATH=%s;%%PATH%%"\n'
                % (dll, mdir, os.path.join(vdir, "Scripts")))
        if pin:
            f.write('set "CUDA_VISIBLE_DEVICES=%s"\n' % pin)

    # 7. use it
    ui.step(7, n, "s_menu")
    print(ui.t("support", u=SUPPORT_URL, r=REPO_URL))
    action = a.action
    pending = None                                      # a newer release this launcher knows of (offered once, then in the chat page)
    if action == "menu":
        print()
        pending = update_check(ui, st, save, src, disabled=check_disabled(a, st))
        if pending:
            rc = apply_update(ui, qs_dir, pending, src, update_answered(a), after_args("menu"), st, save)
            if rc is not None:
                return rc
    while True:
        if action == "menu":
            m = ui.t("menu")
            i = m.rfind("\n")                           # the update item goes after the last item, above the prompt line
            pick = ask(m[:i + 1] + ui.t("menu_update") + m[i + 1:], "6")
            cmd = {"1": "bench", "2": "compare", "3": "web", "4": "chat", "5": "serve", "7": "update"}.get(pick)
        else:
            cmd = None if action == "none" else action
        if not cmd:
            break
        if cmd == "update":
            rel = update_check(ui, st, save, src, force=True)
            if rel:
                rc = apply_update(ui, qs_dir, rel, src, update_answered(a), after_args("menu"), st, save)
                if rc is not None:
                    return rc
            continue
        if cmd == "web":
            rc = run_web(ui, a, vpy, env, mdir, dll, model, pending)
            if rc == UPDATE_RC:                         # the button of the page: that is the yes
                rc = apply_update(ui, qs_dir, pending, src, True, after_args("web", a.web_port_used), st, save)
                if rc != 9:
                    return 0 if rc is None else rc
                pending, rc = None, 0                   # the update failed and nothing changed: back to the menu
            if rc and action != "menu":
                return rc
        elif cmd == "compare":
            script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lookahead_check.py")
            if not os.path.isfile(script):
                ui.say("cmp_nofile", p=script)
                rc = 7
            else:
                print()
                rc = subprocess.call([vpy, script, "--model-dir", mdir, "--dll", dll, "--lang", ui.lang], env=env)
            if rc and action != "menu":
                return rc
        else:
            ui.say("load_note")
            yk(cmd)
        if action != "menu":
            break
    print()
    ui.say("again", f=envfile)
    print(ui.t("support", u=SUPPORT_URL, r=REPO_URL))
    ui.say("bye")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print()
        sys.exit(130)
