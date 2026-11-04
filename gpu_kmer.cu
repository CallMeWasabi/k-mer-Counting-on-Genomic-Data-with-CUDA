#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <vector>
#include <cuda_runtime.h>
#include "gpu_hash_table.hpp"
#include "dna_loader.hpp"

__global__ void gpu_kmer(KeyValue* hashtable, uint64_t capacity, uint64_t* words, uint64_t num_words, uint64_t n,
                         uint32_t k, const uint64_t* breaks, uint64_t num_breaks) {
	uint64_t thread_id = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
	if (n < k || thread_id > n - k) return;

	uint64_t left = 0, right = num_breaks;
	while (left < right) {
		uint64_t mid = (left + right) / 2;
		if (breaks[mid] <= thread_id)
			left = mid + 1;
		else
			right = mid;
	}
	if (left < num_breaks && breaks[left] < thread_id + k) return;

	uint64_t w = thread_id / 32;
	uint64_t s = thread_id & 31;
	uint64_t hi = words[w];
	uint64_t lo = (w + 1 < num_words) ? words[w + 1] : 0;

	uint64_t x;
	if (s == 0)
		x = hi;
	else
		x = (hi << (2 * s)) | (lo >> (64 - 2 * s));

	uint64_t key = x >> (64 - 2 * k);
	uint64_t slot = hash(key, capacity);
	while (true) {
		uint64_t cur = hashtable[slot].key;
		if (cur == key) {
			atomicAdd(&hashtable[slot].val, 1);
			return;
		}
		if (cur == GPU_HASH_EMPTY) {
			uint64_t prev = atomicCAS(&hashtable[slot].key, GPU_HASH_EMPTY, key);
			if (prev == GPU_HASH_EMPTY || prev == key) {
				atomicAdd(&hashtable[slot].val, 1);
				return;
			}
		}

		slot = (slot + 1) & (capacity - 1);
	}
}

int main(int argc, char* argv[]) {
	if (argc < 3) return 1;

	uint32_t k = atoi(argv[2]);
	DNALoader dna = DNALoader(argv[1]);

	uint64_t n = dna.get_dna().length();
	if (n < k) return 0;
	GPUHashTable table(1ull << next_pow2(n * 2));

	// copy to GPU
	auto device_words = make_cuda_unique<uint64_t>(dna.get_words().size());
	cudaMemcpy(device_words.get(), dna.get_words().data(), sizeof(uint64_t) * dna.get_words().size(),
	           cudaMemcpyHostToDevice);

	// positions where a new segment starts after an N run
	auto& breaks = dna.get_breaks();
	auto device_breaks = make_cuda_unique<uint64_t>(breaks.size());
	cudaMemcpy(device_breaks.get(), breaks.data(), sizeof(uint64_t) * breaks.size(), cudaMemcpyHostToDevice);

	int32_t min_grid_size;
	int32_t thread_block_size;
	cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &thread_block_size, gpu_kmer, 0, 0);

	int32_t grid_size = (n - k + thread_block_size) / thread_block_size;
	gpu_kmer<<<grid_size, thread_block_size>>>(table.device_kvs.get(), table.capacity, device_words.get(),
	                                           dna.get_words().size(), n, k, device_breaks.get(), breaks.size());
	cudaDeviceSynchronize();

	// copy the result back: only the non-empty slots
	std::vector<KeyValue> kmers = table.iterate_hashtable();
	uint64_t total = 0;
	for (auto& kv : kmers) total += kv.val;
	printf("k-mer (gpu hash table) distinct: %zu, total: %llu\n", kmers.size(), (unsigned long long)total);

	return 0;
}
