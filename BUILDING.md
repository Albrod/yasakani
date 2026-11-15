# Building yskn.dll from source

The release page has a prebuilt `yskn.dll`. You can also build it yourself. You do not need a GPU to build it, only to run it.
On Linux the same source builds `libyskn.so`: see [Linux](#linux-building-from-source).
日本語：Visual Studio 2022 Build Tools（C++）、CUDA Toolkit、CMake、Ninja を入れ、「x64 Native Tools Command Prompt for VS 2022」で下の 2 つのコマンドを実行します。
Linux（WSL2・Ubuntu 24.04 で確認）では、CUDA Toolkit、GCC、CMake を入れ、下の「Linux (building from source)」の節の手順でソースからビルドします。

## What the release was built with (2026-11-15)

| | |
|---|---|
| OS | Windows 11, x64 |
| C++ compiler | Microsoft Visual C++ 19.44.35226 (Visual Studio 2022 Build Tools, toolset 14.44.35207) |
| CUDA | NVIDIA CUDA Toolkit 13.2 (nvcc 13.2.78) |
| Build tools | CMake (3.24 or newer is required) and Ninja |
| Target | one DLL with machine code for compute capability 8.6 (`sm_86`, RTX 30 series), 8.9 (`sm_89`, RTX 40 series) and 12.0 (`sm_120`, RTX 50 series), plus PTX for `compute_120`. Tested on the RTX 5090 only |

The `yskn.dll` on the release page is this build (SHA-256 `6f8c20644dfa39ed0dea396133a882ecc4451ec19f7926dd737a90540e08e838`). It was built
from this source with the two commands below (the CUDA compiler was named explicitly with `-DCMAKE_CUDA_COMPILER`). Its RTX 50 code is the
code of the build the YASAKANI speed tables in [docs/MEASUREMENTS.md](docs/MEASUREMENTS.md) were measured with (`0f5d973b…0bbb`, 2026-11-07),
plus two fixes (a missing synchronization between GPU threads, and a scratch buffer per engine for the KV5 encoder); the host code also
checks the GPU when it opens a model. In our checks on the RTX 5090 both builds wrote the same tokens. Other compiler or CUDA versions
have not been tried. Compute capability 12.0 needs a recent CUDA Toolkit.

## Steps

Open the **x64 Native Tools Command Prompt for VS 2022** (so that the compiler is on the path), go to the source folder (the one with
`CMakeLists.txt`) and run:

```bat
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

The result is `build\yskn.dll` (about 56 MB; the CUDA runtime is linked statically, so no CUDA Toolkit is needed to run it). The recorded build of
the release took about four minutes and printed many compiler warnings; they are known.

Pass the file to the tools with `--dll`:

```bat
yasakani info --model-dir D:\models\qwen38 --dll build\yskn.dll
```

`yasakani info` prints the SHA-256 of the DLL. A build made with a different compiler or CUDA version will not be byte-identical to the file on the
release page; that is expected. Check the model files with `yasakani verify`, and the whole installation with `yasakani selfcheck`.

## GPUs

By default the build makes machine code for three GPU generations in one DLL: `sm_86` (RTX 30 series), `sm_89` (RTX 40 series) and `sm_120`
(RTX 50 series). Two features need compute capability 9.0 or newer: programmatic dependent launch and the bulk L2 prefetch. The RTX 30 and
RTX 40 code is built without them, and the engine uses them only on a GPU that runs the RTX 50 code. When it opens a model, the engine checks
the GPU and stops with a message on GPUs older than compute capability 8.6 (RTX 20 series and earlier) and on GPUs the build has no code for.
AMD GPUs and Macs are not supported.

Only the RTX 5090 has been tested. The other cards run code compiled from the same source but not tuned for them. They are slower than
the RTX 5090, and we have not checked that the engine works on each of them.

To build for one generation only (a faster build and a smaller file), name it when you configure, in a new build folder:

```bat
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120
```

Use `120` for the RTX 50 series, `89` for the RTX 40 series or `86` for the RTX 30 series. Values below 86 are refused.

One process uses one GPU. On a PC with several GPUs, choose the GPU with the `CUDA_VISIBLE_DEVICES` environment variable.

## Linux (building from source)

The engine also builds on Linux, as `libyskn.so` (contributed in [#3](https://github.com/yasakani-dev/yasakani/pull/3)). This is building from
source only: there is no prebuilt Linux release or launcher, and the quickstart (`START.cmd`) and the optional image input are for Windows.
A prebuilt, officially supported Linux release is one of the project's funding goals.

### What it was tested with (2026-11-14)

| | |
|---|---|
| OS | Ubuntu 24.04 under WSL2 on Windows 11. Native Linux (outside WSL2) has not been tested yet |
| GPU and driver | RTX 5090. Under WSL2 the driver is the Windows NVIDIA driver (tested with 621.24) |
| C++ compiler | GCC 13.3 (the Ubuntu 24.04 default) |
| CUDA | NVIDIA CUDA Toolkit 13.1, 13.3 (nvcc 13.3.73) and 13.5 (nvcc 13.5.37) |
| Build tools | CMake 3.28 (3.24 or newer is required) with its default generator, Unix Makefiles |
| Python | 3.12 (3.10 or newer is required) |

Each of these builds passed `yasakani selfcheck` with the fastest model, and the CUDA 13.5 build with the quality model too. The CUDA 13.5
build followed this section as written, in a new `nvidia/cuda:13.5.0-devel-ubuntu24.04` container (which has the toolkit) with only
`build-essential cmake git python3-venv` added. Other distributions, compilers and CUDA versions have not been tried.

### Before you build

* Install the CUDA Toolkit from NVIDIA's package repository for Ubuntu ([CUDA downloads](https://developer.nvidia.com/cuda-downloads): set up
  the repository, then `sudo apt install cuda-toolkit`, or `cuda-toolkit-13-3` and so on for a given version). Ubuntu's own
  `nvidia-cuda-toolkit` package is CUDA 12.0 on 24.04, too old for compute capability 12.0. Then the other tools:
  `sudo apt install build-essential cmake git python3-venv`.
* Under WSL2, install the toolkit only. The Windows driver is mapped into WSL, and NVIDIA's
  [CUDA on WSL guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html) says not to install a Linux driver there (no `cuda-drivers`
  or `nvidia-open` package). `nvidia-smi` in the WSL terminal should list the RTX 5090.
* Keep the source and the model folder on the Linux file system (for example under `~`), not under `/mnt/c`; files there are slower
  to reach from Linux ([Microsoft's notes](https://learn.microsoft.com/en-us/windows/wsl/filesystems)).

### Steps

Get the source and build:

```sh
git clone https://github.com/yasakani-dev/yasakani
cd yasakani
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
cmake --build build -j
```

`-DCMAKE_CUDA_COMPILER` names the CUDA compiler, which NVIDIA's packages install in `/usr/local/cuda/bin`, a folder that is often not on the
path. The result is `build/bin/libyskn.so` (about 26 MB). The build uses the same CUDA flags as the Windows build (`-O3 --fmad=false`) and
prints some compiler warnings that do not stop it. With `-j` (all compile jobs at once) it took a minute and a quarter on a 16-core CPU and
used at most about 4 GB of memory.

### Running it

Install the Python package in a virtual environment, download a model (here the quality model; `hf` comes with the package's dependencies)
and pass the library with `--dll`:

```sh
python3 -m venv .venv
. .venv/bin/activate
pip install ./python
hf download yasakani-dev/Qwen3.8-27B-YASAKANI-quality --local-dir ~/models/qwen38-quality
yasakani info      --model-dir ~/models/qwen38-quality --dll build/bin/libyskn.so
yasakani verify    --model-dir ~/models/qwen38-quality
yasakani selfcheck --model-dir ~/models/qwen38-quality --dll build/bin/libyskn.so
```

After `pip install ./python` the package does not find `build/bin/libyskn.so` by itself (it looks next to the installed copy, not in the
source folder), so pass `--dll` as above, or set `export YASAKANI_DLL=$PWD/build/bin/libyskn.so` in the source folder. Run from the source
folder instead (`PYTHONPATH=python python -m yasakani selfcheck --model-dir ...`), the package finds it without either.

**The CUDA runtime.** Unlike `yskn.dll`, which has the CUDA runtime linked in statically, `libyskn.so` loads it at run time: it needs
`libcudart.so.13`. The build writes the toolkit's library folder into the file, and NVIDIA's packages also register their library folders with
the system's loader, so on the computer that built it, or on any computer with NVIDIA's CUDA 13 packages, nothing more is needed. Elsewhere,
loading the engine (`selfcheck`, `chat`, `serve`, `bench`; `info` and `verify` do not load it) fails with
`libcudart.so.13: cannot open shared object file`, at the end of a message written for Windows. Then install the CUDA runtime alone from
NVIDIA's repository, ideally the version the library was built with (for example `sudo apt install cuda-cudart-13-5`), or put a folder that holds
`libcudart.so.13` on `LD_LIBRARY_PATH`:

```sh
export LD_LIBRARY_PATH=/usr/local/cuda-13.5/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
```

## Linux

There is no prebuilt Linux release: the engine is built from source, as `libyskn.so`. This is a community-tested path (contributed in
[#3](https://github.com/yasakani-dev/yasakani/pull/3)); the speed tables were measured with the Windows build only.

What a working build was made with (2026-11-14, a contributor's machine):

| | |
|---|---|
| OS | Ubuntu 26.04.1 LTS, x86-64, kernel 7.0.0, glibc 2.43 (native Linux, not WSL) |
| C / C++ compiler | GCC 15.2.0 |
| CUDA | NVIDIA CUDA Toolkit 13.3 (nvcc 13.3.73) |
| Build tools | CMake 4.2.3, Ninja 1.13.2 |
| Python | 3.13.13 for the package (in a virtual environment made with uv 0.11.6) |
| GPU / driver | GeForce RTX 5090, NVIDIA driver 615.71.09 |
| CPU | Intel Core i7-14700K (20 cores, 28 threads) |

You need GCC (or another C++17 compiler), the CUDA Toolkit with `nvcc` on the path, CMake 3.24 or newer and Ninja. Compute capability
12.0 needs a recent CUDA Toolkit. In the source folder run:

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

The result is `build/bin/libyskn.so` (about 26 MB). Unlike `yskn.dll`, it loads the CUDA runtime (`libcudart.so`) as a shared
library, so the machine that runs it needs that library too (it comes with the CUDA Toolkit). Install the Python package and point it at
the library with `--dll` or with the environment variable `YASAKANI_DLL`:

```sh
python3 -m venv venv
venv/bin/pip install ./python
export YASAKANI_DLL=$PWD/build/bin/libyskn.so
venv/bin/yasakani info --model-dir /models/qwen38-speed
venv/bin/yasakani verify --model-dir /models/qwen38-speed
venv/bin/yasakani selfcheck --model-dir /models/qwen38-speed
venv/bin/yasakani serve --model-dir /models/qwen38-speed
```

When the package is run from the source tree, `build/bin/libyskn.so` is also found without `--dll`.

### Image input on Linux

The optional image input ([python/README.md](python/README.md#image-input-optional)) uses the multimodal library of llama.cpp b10488. On
Windows the server takes the libraries of the b10488 release and compares them by SHA-256. That release has no matching Linux package to
compare with, so on Linux the same release is built from source, as shared libraries:

```sh
git clone --depth 1 --branch b10488 https://github.com/ggml-org/llama.cpp llama-b10488-src
cmake -S llama-b10488-src -B llama-b10488-build -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_SHARED_LIBS=ON -DGGML_CUDA=ON -DLLAMA_CURL=OFF \
      -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_TOOLS=ON \
      -DCMAKE_INSTALL_RPATH='$ORIGIN' -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON
cmake --build llama-b10488-build --target mtmd
mkdir llama-b10488
cp -a llama-b10488-build/bin/lib*.so* llama-b10488/
```

* `-DLLAMA_BUILD_TOOLS=ON` is needed because the `mtmd` target belongs to the tools; `--target mtmd` then builds only the library and
  what it depends on, not the programs.
* The `RPATH` options let the libraries find each other in whatever folder they are copied to, with no `LD_LIBRARY_PATH`.
* `-DGGML_CUDA=ON` is only needed for the projector on the GPU (below). A CUDA build was the one tested, in both modes; a build
  without it should be enough for the CPU mode, but that has not been tried. The CUDA backend (`libggml-cuda.so`) needs the CUDA
  runtime libraries (`libcudart`, `libcublas`) where the loader finds them; a backend that cannot be loaded is skipped without an
  error, and the projector then runs on the CPU.

Pass the folder as `--llama-dir`, with `--mmproj` and `--vision-model` as on Windows, and install the package with Pillow
(`pip install "./python[vision]"`):

```sh
venv/bin/yasakani serve --model-dir /models/qwen38-quality \
  --mmproj /gguf/mmproj-Qwen3.8-27B-BF16.gguf --llama-dir $PWD/llama-b10488 --vision-model /gguf/Qwen3.8-27B-Q4_K_M.gguf
```

On Linux the server checks only that `libllama.so`, `libmtmd.so`, `libggml.so` and `libggml-base.so` are in the folder; there is no
checksum, so it is up to you to build exactly b10488. A library with another structure layout is still refused when the encoder opens.

By default the projector runs on the CPU, as on Windows. `YASAKANI_VISION_GPU=1` in the environment of the server moves it to the GPU,
with flash attention switched off; the measurements behind that choice, and its memory cost, are in
[python/README.md](python/README.md#image-input-on-linux). Mind the GPU memory: the projector's work memory comes on top of the model.

## Layout

* `src/`: the engine (C++ and CUDA); `include/`: the C interface that the Python package binds to; `third_party/`: nlohmann/json (MIT).
* The Python package is built and installed separately (`pip install .` in its folder).
* `tools/convert/`: the program that made the released model files from the original weights (Qwen3.8-27B only). See
  [tools/convert/README.md](tools/convert/README.md).
