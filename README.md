# Large-Scale k-mer Counting on Genomic Data with CUDA

- By 6710451321 Suphachock Kamlek 

Counts every k-mer (substring of length k) in a DNA sequence with an open-addressing hash table on the GPU,
and compares it against the CPU (`std::unordered_map`) on the *E. coli* K-12 genome and human chromosome 1 (GRCh38).

## Requirements

| Tool | Tested version | Notes |
|---|---|---|
| NVIDIA GPU | GeForce RTX 5060 Ti 16 GB (sm_120) | For other cards, change `ARCH` (see below) |
| CUDA Toolkit | 13.4 | `nvcc` must be on `PATH` |
| Visual Studio (MSVC) | 2026 Community (v18) | On Windows, `nvcc` uses `cl.exe` as the host compiler |
| GNU Make | 3.81 (GnuWin32) or `mingw32-make` | |

On Windows, `nvcc` needs the MSVC environment.
The Makefile calls `nvcc` through [tools/nvcc.bat](tools/nvcc.bat), which loads `vcvars64.bat` automatically.
If Visual Studio is installed somewhere else, set `VCVARS` before running make:

```powershell
$env:VCVARS = "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
```

## Datasets

Place the files in `data/`.

| File | Size | Source | Used for |
|---|---|---|---|
| `ecoli_k12_mg1655.fa` | 4.6 M bases | NCBI [NC_000913.3](https://www.ncbi.nlm.nih.gov/nuccore/NC_000913.3) | Correctness checks, fast runs |
| `grch38_chr1.fa` | 249 M bases (18.5 M `N`) | Ensembl [GRCh38 `dna`](https://ftp.ensembl.org/pub/current_fasta/homo_sapiens/dna/) chromosome 1 | Performance measurements |

## Quick start

```bash
make          # build every program into bin/
make demo     # count the k-mers of E. coli on the GPU and check against the CPU
```

Expected output of `make demo` (k = 31):

```
k-mer (gpu hash table) distinct: 4570839, total: 4641622
...
result : CPU distinct 4570839 total 4641622 | GPU distinct 4570839 total 4641622 -> MATCH
```

## Make targets

| Command | What it does | Approx. time |
|---|---|---|
| `make` / `make all` | Build every program into `bin/` | ~1 min |
| `make demo` | `gpu_kmer` on E. coli + CPU vs GPU result and timing check | a few seconds |
| `make demo-cpu` | `cpu_with_pack` on E. coli, writes every k-mer and its count to `out/cpu_kmers.txt` | a few seconds |
| `make bench` | CPU vs GPU on E. coli (10 GPU runs, 3 CPU runs) | a few seconds |
| `make bench-chr1` | CPU vs GPU on chr1 | **~8 min** (the CPU part takes ~7 min) |
| `make bench-table` | GPU hash table with random keys: insert/lookup at load factor 0.1–0.9 vs `std::unordered_map` | ~1 min |
| `make bench-all` | Run every benchmark above | ~10 min |
| `make clean` | Delete `bin/` and `out/` | |

Variables:

| Variable | Default | Meaning |
|---|---|---|
| `K` | `31` | k-mer length (1–31) |
| `REPS` | `10` | Number of GPU runs in `make bench` |
| `ECOLI` | `data/ecoli_k12_mg1655.fa` | Input for the demo and small benchmarks |
| `CHR1` | `data/grch38_chr1.fa` | Input for the large benchmarks |
| `ARCH` | `sm_120` | GPU compute capability, e.g. `sm_89` (RTX 40), `sm_86` (RTX 30) |

```bash
make bench K=25 REPS=20
make demo ECOLI=data/my_genome.fa
make ARCH=sm_89
```

## Building without make

Run in PowerShell from the project folder (`tools\nvcc.bat` loads the MSVC environment):

```powershell
tools\nvcc.bat -O3 -std=c++17 -arch=sm_120 -rdc=true gpu_kmer.cu gpu_hash_table.cu dna_loader.cpp -o bin/gpu_kmer.exe
bin\gpu_kmer.exe data/ecoli_k12_mg1655.fa 31
```

## Reference results (RTX 5060 Ti, k = 31)

| Dataset | CPU `unordered_map` | GPU kernel | GPU end-to-end (H2D + reset + kernel + D2H) |
|---|---|---|---|
| E. coli (4.6 M k-mers) | ~770 ms | 3.7 ms | 25.7 ms |
| chr1 (230 M k-mers) | ~437 s | 211.6 ms | 1,177 ms |

Both datasets match the CPU result (chr1: distinct 208,061,649, total 230,476,809).
Numbers vary by machine; run `make bench` and `make bench-chr1` to measure on yours.

## Limitations

- The whole table must fit in GPU memory: chr1 uses 2²⁹ slots × 16 B = 8.6 GB
- k ≤ 31, because `0xFFFF…FF` is the empty-slot marker
- No canonical counting yet (a k-mer and its reverse complement are counted separately)
- The Makefile supports Windows only (recipes run in `cmd.exe`)

## Documents

- [slides/kmer_cuda_slides.pdf](slides/kmer_cuda_slides.pdf): presentation slides
