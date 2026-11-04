#include <chrono>
#include <cstdio>
#include <random>
#include <unordered_map>
#include <vector>
#include "gpu_hash_table.hpp"

const int REPS = 5;
const double LOAD_FACTORS[] = {0.1, 0.25, 0.5, 0.75, 0.9};

template <typename Kernel> int block_size(Kernel kernel) {
	int min_grid_size, thread_block_size;
	cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &thread_block_size, kernel, 0, 0);
	return thread_block_size;
}

// average kernel time over REPS runs, `setup` is not timed
template <typename S, typename F> double time_gpu(S setup, F work) {
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	double total = 0;
	for (int i = 0; i < REPS; i++) {
		setup();
		cudaEventRecord(start);
		work();
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		float ms;
		cudaEventElapsedTime(&ms, start, stop);
		total += ms;
	}
	return total / REPS;
}

template <typename F> double time_cpu(F work) {
	auto start = std::chrono::high_resolution_clock::now();
	work();
	auto end = std::chrono::high_resolution_clock::now();
	return std::chrono::duration<double, std::milli>(end - start).count();
}

struct Result {
	double load_factor;
	size_t n;
	double kernel_insert, kernel_lookup, e2e_insert, e2e_lookup, cpu_insert, cpu_lookup; // ms
};

double mops(size_t n, double ms) { return n / ms / 1000.0; }

int main(int argc, char* argv[]) {
	const size_t max_n = (1 << 24) * 2 * 0.9; // enough keys for the highest load factor of the 2^25-slot table

	std::mt19937_64 rnd(42);
	std::uniform_int_distribution<uint64_t> dis(0, GPU_HASH_EMPTY - 1);
	std::vector<KeyValue> kvs(max_n);
	for (auto& kv : kvs) kv = {dis(rnd), dis(rnd)};

	GPUHashTable table((1 << 24) * 2);
	auto device_input_kvs = make_cuda_unique<KeyValue>(max_n);
	cudaMemcpy(device_input_kvs.get(), kvs.data(), sizeof(KeyValue) * max_n, cudaMemcpyHostToDevice);

	int insert_block = block_size(gpu_hashtable_insert);
	int lookup_block = block_size(gpu_hashtable_lookup);
	size_t n = 0;
	auto clear = [&] { cudaMemset(table.device_kvs.get(), 0xff, sizeof(KeyValue) * table.capacity); };
	auto insert = [&] {
		gpu_hashtable_insert<<<(n + insert_block - 1) / insert_block, insert_block>>>(
		    table.device_kvs.get(), table.capacity, device_input_kvs.get(), n);
	};
	auto lookup = [&] {
		gpu_hashtable_lookup<<<(n + lookup_block - 1) / lookup_block, lookup_block>>>(
		    table.device_kvs.get(), table.capacity, device_input_kvs.get(), n);
	};

	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	printf("GPU      : %s (%d SMs)\n", prop.name, prop.multiProcessorCount);
	printf("Table    : %llu slots x %zu B = %.0f MiB\n", table.capacity, sizeof(KeyValue),
	       table.capacity * sizeof(KeyValue) / 1048576.0);
	printf("Block    : insert %d, lookup %d threads\n", insert_block, lookup_block);
	printf("GPU reps : %d (average), CPU reps: 1\n\n", REPS);

	n = max_n; // warm-up
	insert();
	lookup();

	std::vector<Result> results;
	for (double lf : LOAD_FACTORS) {
		n = table.capacity * lf;
		std::vector<KeyValue> sub(kvs.begin(), kvs.begin() + n);
		Result r{lf, n};
		printf("Running load factor %.2f (%zu keys)...\n", lf, n);

		// GPU kernel only: data already on device
		r.kernel_insert = time_gpu(clear, insert);
		r.kernel_lookup = time_gpu([] {}, lookup);

		// GPU end-to-end: cudaMalloc + H2D copy + kernel (+ D2H for lookup)
		clear();
		cudaDeviceSynchronize();
		r.e2e_insert = time_cpu([&] {
			table.insert(sub);
			cudaDeviceSynchronize();
		});
		r.e2e_lookup = time_cpu([&] { table.lookup(sub); });

		// CPU baseline: std::unordered_map, single thread
		std::unordered_map<uint64_t, uint64_t> map;
		map.reserve(n);
		volatile uint64_t sum = 0;
		r.cpu_insert = time_cpu([&] {
			for (auto& kv : sub) map[kv.key] = kv.val;
		});
		r.cpu_lookup = time_cpu([&] {
			for (auto& kv : sub) sum += map.find(kv.key)->second;
		});

		results.push_back(r);
	}

	printf("\n[1] Time (ms)\n");
	printf("%-5s %10s | %9s %9s | %9s %9s | %9s %9s\n", "load", "keys", "kern ins", "kern lkp", "e2e ins", "e2e lkp",
	       "cpu ins", "cpu lkp");
	for (auto& r : results)
		printf("%-5.2f %10zu | %9.3f %9.3f | %9.3f %9.3f | %9.1f %9.1f\n", r.load_factor, r.n, r.kernel_insert,
		       r.kernel_lookup, r.e2e_insert, r.e2e_lookup, r.cpu_insert, r.cpu_lookup);

	printf("\n[2] Throughput (M ops/s)\n");
	printf("%-5s %10s | %9s %9s | %9s %9s | %9s %9s\n", "load", "keys", "kern ins", "kern lkp", "e2e ins", "e2e lkp",
	       "cpu ins", "cpu lkp");
	for (auto& r : results)
		printf("%-5.2f %10zu | %9.1f %9.1f | %9.1f %9.1f | %9.1f %9.1f\n", r.load_factor, r.n,
		       mops(r.n, r.kernel_insert), mops(r.n, r.kernel_lookup), mops(r.n, r.e2e_insert), mops(r.n, r.e2e_lookup),
		       mops(r.n, r.cpu_insert), mops(r.n, r.cpu_lookup));

	printf("\n[3] GPU vs CPU\n");
	auto compare = [](const char* name, size_t n, double gpu_ms, double cpu_ms) {
		printf("  %-14s GPU %8.3f ms (%7.1f M ops/s)  vs  CPU %8.1f ms (%5.1f M ops/s)  ->  %4.0fx faster\n", name,
		       gpu_ms, mops(n, gpu_ms), cpu_ms, mops(n, cpu_ms), cpu_ms / gpu_ms);
	};
	for (auto& r : results) {
		printf("load %.2f (%zu keys)\n", r.load_factor, r.n);
		compare("insert kernel", r.n, r.kernel_insert, r.cpu_insert);
		compare("lookup kernel", r.n, r.kernel_lookup, r.cpu_lookup);
		compare("insert e2e", r.n, r.e2e_insert, r.cpu_insert);
		compare("lookup e2e", r.n, r.e2e_lookup, r.cpu_lookup);
	}

	return 0;
}

/*
    nvcc -O3 -std=c++17 -arch=sm_120 gpu_hash_table_benchmark.cu -o gpu_hash_table_benchmark.exe &&
   ./gpu_hash_table_benchmark.exe
*/
