// CPU (std::unordered_map) vs GPU (gpu_kmer) k-mer counting benchmark
//
// build: nvcc -O3 -std=c++17 -arch=sm_120 gpu_kmer_benchmark.cu dna_loader.cpp -o gpu_kmer_benchmark.exe
// run  : gpu_kmer_benchmark.exe <fasta> <k> [gpu_reps=10] [cpu_reps=1]
//
// Reading and packing the FASTA file is shared and not timed.

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <unordered_map>
#include <vector>
#include <cuda_runtime.h>

#define main gpu_kmer_main
#include "gpu_kmer.cu"
#undef main
#include "gpu_hash_table.cu"

struct Count {
	uint64_t distinct, total;
};

// same algorithm as cpu_with_pack.cpp: rolling key, restart after each N run
Count cpu_kmer(const std::vector<uint64_t>& words, uint64_t n, const std::vector<uint64_t>& breaks, uint32_t k) {
	std::unordered_map<uint64_t, uint64_t> mp;
	mp.reserve(n);
	uint64_t mask = (k == 32) ? ~0ull : (1ull << (2 * k)) - 1;
	uint64_t key = 0, run = 0, total = 0;
	size_t b = 0;
	for (uint64_t i = 0; i < n; i++) {
		if (b < breaks.size() && breaks[b] == i) {
			key = 0;
			run = 0;
			b++;
		}
		key = ((key << 2) | ((words[i / 32] >> (62 - 2 * (i & 31))) & 3)) & mask;
		if (++run >= k) {
			mp[key]++;
			total++;
		}
	}
	return {mp.size(), total};
}

void row(const char* name, std::vector<double> ms) {
	std::sort(ms.begin(), ms.end());
	printf("  %-26s %12.3f %12.3f %12.3f\n", name, ms[ms.size() / 2], ms.front(), ms.back());
}

double median(std::vector<double> v) {
	std::sort(v.begin(), v.end());
	return v[v.size() / 2];
}

int main(int argc, char* argv[]) {
	if (argc < 3) {
		printf("usage: gpu_kmer_benchmark <fasta> <k> [gpu_reps=10] [cpu_reps=1]\n");
		return 1;
	}
	uint32_t k = atoi(argv[2]);
	int gpu_reps = argc > 3 ? atoi(argv[3]) : 10;
	int cpu_reps = argc > 4 ? atoi(argv[4]) : 1;

	DNALoader dna(argv[1]);
	auto& words = dna.get_words();
	auto& breaks = dna.get_breaks();
	uint64_t n = dna.get_dna().length();
	if (n < k) {
		printf("sequence shorter than k\n");
		return 1;
	}

	// CPU
	Count cpu{};
	std::vector<double> cpu_ms;
	for (int r = 0; r < cpu_reps; r++) {
		auto t0 = std::chrono::steady_clock::now();
		cpu = cpu_kmer(words, n, breaks, k);
		cpu_ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
	}

	// GPU
	GPUHashTable table(1ull << next_pow2(n * 2));
	auto device_words = make_cuda_unique<uint64_t>(words.size());
	auto device_breaks = make_cuda_unique<uint64_t>(breaks.size());

	int min_grid_size, block;
	cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block, gpu_kmer, 0, 0);
	uint64_t grid = (n - k + block) / block;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	auto gpu_time = [&](auto work) {
		cudaEventRecord(start);
		work();
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		float ms;
		cudaEventElapsedTime(&ms, start, stop);
		return (double)ms;
	};

	Count gpu{};
	std::vector<double> h2d_ms, reset_ms, kernel_ms, d2h_ms, e2e_ms;
	for (int r = -1; r < gpu_reps; r++) { // r = -1: warm-up, not recorded
		double h2d = gpu_time([&] {
			cudaMemcpy(device_words.get(), words.data(), sizeof(uint64_t) * words.size(), cudaMemcpyHostToDevice);
			cudaMemcpy(device_breaks.get(), breaks.data(), sizeof(uint64_t) * breaks.size(), cudaMemcpyHostToDevice);
		});
		double reset = gpu_time([&] {
			gpu_hashtable_reset<<<(table.capacity + 255) / 256, 256>>>(table.device_kvs.get(), table.capacity);
		});
		double kernel = gpu_time([&] {
			gpu_kmer<<<grid, block>>>(table.device_kvs.get(), table.capacity, device_words.get(), words.size(), n, k,
			                          device_breaks.get(), breaks.size());
		});
		std::vector<KeyValue> kmers;
		double d2h = gpu_time([&] { kmers = table.iterate_hashtable(); });
		if (r < 0) continue;

		uint64_t total = 0;
		for (auto& kv : kmers) total += kv.val;
		gpu = {kmers.size(), total};
		h2d_ms.push_back(h2d);
		reset_ms.push_back(reset);
		kernel_ms.push_back(kernel);
		d2h_ms.push_back(d2h);
		e2e_ms.push_back(h2d + reset + kernel + d2h);
	}

	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	printf("input  : %s, %llu bases (N removed), %zu segments, k = %u\n", argv[1], (unsigned long long)n,
	       breaks.size() + 1, k);
	printf("GPU    : %s, grid %llu x block %d\n", prop.name, (unsigned long long)grid, block);
	printf("table  : %llu slots x %zu B = %.2f GB\n", (unsigned long long)table.capacity, sizeof(KeyValue),
	       table.capacity * sizeof(KeyValue) / 1e9);

	bool match = cpu.distinct == gpu.distinct && cpu.total == gpu.total;
	printf("\nresult : CPU distinct %llu total %llu | GPU distinct %llu total %llu -> %s\n",
	       (unsigned long long)cpu.distinct, (unsigned long long)cpu.total, (unsigned long long)gpu.distinct,
	       (unsigned long long)gpu.total, match ? "MATCH" : "MISMATCH");

	printf("\ntime (ms)                        median          min          max\n");
	printf("CPU, %d run(s)\n", cpu_reps);
	row("unordered_map count", cpu_ms);
	printf("GPU, %d run(s) after 1 warm-up\n", gpu_reps);
	row("H2D words + breaks", h2d_ms);
	row("table reset", reset_ms);
	row("kernel", kernel_ms);
	row("D2H iterate_hashtable", d2h_ms);
	row("end-to-end (sum)", e2e_ms);

	double cpu_t = median(cpu_ms), kernel_t = median(kernel_ms), e2e_t = median(e2e_ms);
	auto mkmers = [&](double ms) { return cpu.total / ms / 1e3; };
	printf("\nthroughput (M k-mers/s): CPU %.1f | GPU kernel %.1f | GPU end-to-end %.1f\n", mkmers(cpu_t),
	       mkmers(kernel_t), mkmers(e2e_t));
	printf("speedup  (CPU / GPU)  : kernel only %.1fx | end-to-end %.1fx\n", cpu_t / kernel_t, cpu_t / e2e_t);
	return match ? 0 : 1;
}
