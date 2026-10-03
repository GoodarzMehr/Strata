# Unsloth Q4_K_XL on two RTX 4090s, FP16 KV

Measured locally on 2026-10-02: Ubuntu 22.04.5, Ryzen 9950X (16 cores, 32 threads, AVX-512),
96 GB RAM (about 66 GB available before loading), two RTX 4090s with 24 GB each, NVIDIA driver
610.57.04 and an NVMe model drive. The GPUs communicate across a PCIe host bridge; no NVLink.
Engine 0.1.36 was built locally for SM 89 with CUDA 13.0.88 and GCC/G++ 11. The optional
`STRATA_MMQ_KQUANTS` prompt kernels were left off: they introduce INT8 activation rounding.
The base source commit is `36fa455`, with the development changes for this setup. Build metadata:
[BUILD.json](BUILD.json).

## Model and settings

The existing merged `Qwen3.8-Flash-Next-UD-Q4_K_XL.gguf` is 103.69 GiB. Its metadata reports
architecture `qwen4exp` and native context 262144. The directory and tensor ranges were checked;
the whole merged file was not SHA256-verified against a published merged-file hash. The routed
experts and PLE table are mapped from that file, with no `experts.bin` copy. Each card owns
24 of the 48 layers. Experts missing from the GPU caches run on the CPU using the OS file cache.

`mtp-Qwen3.8-Flash-Next-BF16.gguf` supplied the MTP tensors. All 31 reconstructed tensors matched
the pinned checkpoint SHA256 before packing into Strata's usual Q2 draft format. Draft proposals
are verified by the Q4 target. The full and shared BF16 MTP forms are accepted by the importer;
quantized or mismatched MTP files are refused.

The measured configuration uses:

```text
--expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5
--kv fp16 --kv-resident 32768 --mmap-experts --pcie-frac 0
--vram-reserve-mib 1280 --stats
```

FP16 K/V is held in pinned host RAM, with selected attention blocks fetched into the GPU's
32768-position resident cache. The complete configured context is retained. At 262144, main
and draft K/V alone need about 6.5 GiB of host RAM; indexer state, checkpoints and buffers are extra.
The model uses its trained sparse attention and recurrent layers. No RoPE scaling or experimental
speed projection was enabled.

The default 700 MiB VRAM reserve failed: the original 262144 run stopped in cuBLAS prompt processing,
and a 200000 run with a 4095-token prompt ran out of memory creating a decode graph. Increasing
the reserve to 1280 MiB fixed both the 200000 and native 262144 runs. Native 262144 was left configured;
200000 is the verified fallback if other applications need more RAM.

## Context and recall measurements

These are individual requests, greedy with thinking disabled and 192 output tokens. No prompt tokens
were reused. Each long prompt contains three markers at roughly 10%, 50% and 90% of its character
length, followed by a request to quote them and explain an LRU cache. The local tokenizer and server
agreed on the input count. All three markers were returned in each successful long test.

| Context | Input tokens | Prompt seconds | Prompt tok/s | First token seconds | Decode tok/s | Expert hits | Markers |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 200000 | 4095 | 12.34 | 331.8 | 12.39 | 40.5 | 89.6% | 3/3 |
| 200000 | 199799 | 133.48 | 1496.8 | 133.90 | 62.5 | 94.6% | 3/3 |
| 262144 | 4096 | 13.84 | 296.1 | 13.89 | 37.0 | 89.5% | 3/3 |
| 262144 | 261899 | 156.75 | 1670.8 | 157.29 | 57.9 | 94.7% | 3/3 |

The 4096-token request ran first after each restart, then the near-full-context request. The file cache
was not flushed. Prompt time includes reading experts and restoring lent GPU cache slots; decode rate
excludes prompt processing. The answers reached the output limit after quoting all markers. This is
a small recall check, not a broad model quality evaluation. Cache temperature, routing and draft
acceptance affect speed; these figures are not a universal output rate. In plain mmap mode, the engine's
`file_mb` counter describes accessed mapped bytes, not measured physical SSD traffic.

Raw API usage, timings, answers and request hashes: [results.json](results.json).

## Calibration

`tools/calibrate.py` ran three short prompts per setting (list merging, refrigerator operation and
European capitals), greedy with 128 output tokens, with warm-up and interleaved remeasurement.
It took 434 seconds. The winning draft confidence floor was `--spec-min-p 0.70`; the repeated
candidate median was 57.9 tok/s versus 53.2 for the default. The candidate also differed in the
PCIe fraction, so this is not an isolated measurement of the confidence floor.

The CPU worker sweep kept 15 workers: medians were 52.0, 50.5 and 50.4 tok/s for 15, 10 and 8
workers. The final configuration keeps `--pcie-frac 0`: directly mapped, unpinned expert misses
run on the CPU. It uses the measured confidence floor 0.70 and the default worker count.
Full sweep and repeated rates: [calibration.json](calibration.json). These short workloads do not
establish the fastest settings for every prompt.

After restarting with those final settings, a 58-token LRU coding prompt generated 192 tokens at
30.1 tok/s (first token 2.33 seconds). A subsequent 261900-token prompt retrieved all three markers
and generated 192 tokens at 49.4 tok/s (prompt processing 161.45 seconds, 1622.2 tok/s; first token
161.98 seconds). Both appear in `results.json`. The long-context result is slower than the earlier
default-floor run; its different prompt, preceding workload, routing and draft windows make this
an end-to-end validation, not an isolated drafting comparison. No setting is universally fastest.

One sample during the final long prompt showed 23198 MiB used on GPU 0 and 23396 MiB on GPU 1,
out of 24564 MiB each. Ubuntu reported about 53 GiB of available RAM, including reclaimable file
cache, with no swap. These are samples, not a measured peak or a memory guarantee under other loads.

## Reproduce

From the repository root, use the local files with the
[setup command](../../../docs/UNSLOTH_Q4.md#local-gguf-two-gpus-and-fp16-kv). This machine's default
`g++` is too old for C++20; a clean build must select the installed GCC/G++ 11 or 12 explicitly.
For GCC 12, prefix the setup command with
`CC=/usr/bin/gcc-12 CXX=/usr/bin/g++-12 CUDAHOSTCXX=/usr/bin/g++-12` and add `--build`.
The measured build used GCC 11, already saved in its CMake cache.

Start the server on localhost, then run:

```sh
.venv/bin/python bench/results/2026-10-02-dual-4090-q4-fp16/benchmark.py \
  --tokens 4096 --out small.json
.venv/bin/python bench/results/2026-10-02-dual-4090-q4-fp16/benchmark.py \
  --tokens 261900 --out native-context.json
```

Use `--config PATH` or `--url URL` for another local configuration. The script requires room for 192
output tokens plus a small margin. With context 200000 use `--tokens 199800`. It builds filler from
the current repository, so the exact prompt changes with source edits and its nonce. A fresh request
after a restart is slower than one with warm file pages.

To change the window, edit the value following `--max-context` in
`strata-unsloth-ud-q4_k_xl.json` and restart. Keep `--kv fp16`, `--kv-resident 32768` and the VRAM
reserve. Context includes prompt and output. Going past 262144 needs experimental RoPE scaling
and has not been validated here.

## Accuracy and multiple agents

FP16 KV avoids the optional INT8 and Q4 KV rounding. Unsloth's required `--compat-bf16` conversion
still rounds 195 small Q8_0 projections to BF16; the PLE convolution is loaded as FP16. CPU and GPU
expert paths can also round differently. This setup does not promise identical logits or answers
to llama.cpp on the same Q4 file, or accuracy equal to the original unquantized checkpoint.

Multiple API clients share the loaded weights, but generation is serialized. Full conversation
parking is incompatible with the GPU layer split; ordinary prefix checkpoints still work.
Simultaneous generation with a unified multi-sequence KV pool, as with llama.cpp's `-np 2 -kvu`,
has not been implemented.

## Validation

- 167 setup tests and 9 local MTP import tests passed without GPU access or downloads.
- 14 calibration tests and 100 server tests passed (the latter require localhost sockets).
- 10 CUDA tests passed: FP16 streaming parity, native quantized expert parity, QSA parity and
  conversation-cache state checks. Streaming parity was bitwise for the stored FP16 cells.
- Real dual-GPU API requests completed at both context 200000 and 262144, including near-full windows.
