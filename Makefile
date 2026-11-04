# k-mer counting on genomic data with CUDA
#
#   make                 build every program into bin/
#   make demo            quick demo on E. coli: GPU count + CPU vs GPU check
#   make demo-cpu        CPU program, writes every k-mer and its count to out/
#   make bench           CPU vs GPU benchmark on E. coli (seconds)
#   make bench-chr1      CPU vs GPU benchmark on chr1 (CPU part takes ~7 minutes)
#   make bench-table     GPU hash table insert/lookup vs load factor (random keys)
#   make bench-all       every benchmark above
#   make clean
#
# variables: K=31  ECOLI=data/ecoli_k12_mg1655.fa  CHR1=data/grch38_chr1.fa  REPS=10  ARCH=sm_120
#   e.g.  make bench K=25 REPS=20

K      ?= 31
REPS   ?= 10
ARCH   ?= sm_120
ECOLI  ?= data/ecoli_k12_mg1655.fa
CHR1   ?= data/grch38_chr1.fa

NVFLAGS := -O3 -std=c++17 -arch=$(ARCH)

BIN := bin
OUT := out

# Windows only (nvcc needs MSVC): every recipe runs in cmd.exe, whether make is started
# from PowerShell, cmd or Git Bash
SHELL  := cmd.exe
MKDIR   = if not exist $(subst /,\,$(1)) mkdir $(subst /,\,$(1))
RMDIR   = if exist $(subst /,\,$(1)) rmdir /s /q $(subst /,\,$(1))
NVCC   := tools\nvcc.bat
EXE     = $(BIN)\$(1).exe

PROGRAMS := gpu_kmer gpu_kmer_benchmark gpu_hash_table_benchmark cpu_with_pack
TARGETS  := $(addprefix $(BIN)/,$(addsuffix .exe,$(PROGRAMS)))

COMMON_HDR := dna_loader.hpp gpu_hash_table.hpp

.PHONY: all demo demo-cpu bench bench-chr1 bench-table bench-all clean help

all: $(TARGETS)

help:
	@echo targets: all demo demo-cpu bench bench-chr1 bench-table bench-all clean

# ---------------------------------------------------------------- build

$(BIN):
	@$(call MKDIR,$(BIN))

$(OUT):
	@$(call MKDIR,$(OUT))

# GPU k-mer counter: one thread per position, open-addressing table in GPU memory
# -rdc=true: the kernel in gpu_kmer.cu calls the __device__ hash() defined in gpu_hash_table.cu
$(BIN)/gpu_kmer.exe: gpu_kmer.cu gpu_hash_table.cu dna_loader.cpp $(COMMON_HDR) | $(BIN)
	$(NVCC) $(NVFLAGS) -rdc=true gpu_kmer.cu gpu_hash_table.cu dna_loader.cpp -o $@

# #includes gpu_kmer.cu and gpu_hash_table.cu itself
$(BIN)/gpu_kmer_benchmark.exe: gpu_kmer_benchmark.cu gpu_kmer.cu gpu_hash_table.cu dna_loader.cpp $(COMMON_HDR) | $(BIN)
	$(NVCC) $(NVFLAGS) gpu_kmer_benchmark.cu dna_loader.cpp -o $@

$(BIN)/gpu_hash_table_benchmark.exe: gpu_hash_table_benchmark.cu gpu_hash_table.cu gpu_hash_table.hpp | $(BIN)
	$(NVCC) $(NVFLAGS) gpu_hash_table_benchmark.cu gpu_hash_table.cu -o $@

# CPU program; next_pow2() lives in gpu_hash_table.cu, so it is linked in too
$(BIN)/cpu_with_pack.exe: cpu_with_pack.cpp gpu_hash_table.cu dna_loader.cpp $(COMMON_HDR) | $(BIN)
	$(NVCC) $(NVFLAGS) cpu_with_pack.cpp gpu_hash_table.cu dna_loader.cpp -o $@

# ---------------------------------------------------------------- demo

demo: $(BIN)/gpu_kmer.exe $(BIN)/gpu_kmer_benchmark.exe
	@echo === GPU k-mer count: $(ECOLI), k = $(K)
	$(call EXE,gpu_kmer) $(ECOLI) $(K)
	@echo === CPU vs GPU, correctness + timing
	$(call EXE,gpu_kmer_benchmark) $(ECOLI) $(K) 5 1

demo-cpu: $(BIN)/cpu_with_pack.exe | $(OUT)
	@echo === CPU k-mer count: $(ECOLI), k = $(K), result in $(OUT)/cpu_kmers.txt
	$(call EXE,cpu_with_pack) $(ECOLI) $(K) > $(OUT)\cpu_kmers.txt

# ---------------------------------------------------------------- benchmarks

bench: $(BIN)/gpu_kmer_benchmark.exe
	$(call EXE,gpu_kmer_benchmark) $(ECOLI) $(K) $(REPS) 3

bench-chr1: $(BIN)/gpu_kmer_benchmark.exe
	@echo === chr1: the CPU std::unordered_map baseline alone takes about 7 minutes
	$(call EXE,gpu_kmer_benchmark) $(CHR1) $(K) 5 1

bench-table: $(BIN)/gpu_hash_table_benchmark.exe
	$(call EXE,gpu_hash_table_benchmark)

bench-all: bench bench-table bench-chr1

# ---------------------------------------------------------------- clean

clean:
	@$(call RMDIR,$(BIN))
	@$(call RMDIR,$(OUT))
