#pragma once
#include <cstdint>
#include <driver_types.h>
#include <memory>
#include <vector>
#include <cuda_runtime.h>

/*
Implement GPU Hash table with condition
1. static hash table (size of table)
2. no opts remove element
3. no opts rehash
*/

struct CudaDeleter {
	void operator()(void* p) const noexcept {
		if (p) cudaFree(p); // no throw in deleter
	}
};

template <typename T> using cuda_unique_ptr = std::unique_ptr<T[], CudaDeleter>;

template <typename T> using cuda_shared_ptr = std::shared_ptr<T[]>;

template <typename T> cuda_unique_ptr<T> make_cuda_unique(size_t n) {
	T* raw = nullptr;
	cudaMalloc(&raw, n * sizeof(T));
	return cuda_unique_ptr<T>(raw);
}

template <typename T> cuda_shared_ptr<T> make_cuda_shared(size_t n) {
	T* raw = nullptr;
	cudaMalloc(&raw, n * sizeof(T));
	return cuda_shared_ptr<T>(raw, [](T* p) { cudaFree(p); });
}

struct alignas(16) KeyValue {
	uint64_t key;
	uint64_t val;
};

const uint64_t GPU_HASH_EMPTY = 0xffffffffffffffff;

class GPUHashTable {
public:
	cuda_unique_ptr<KeyValue> device_kvs;
	uint64_t capacity;
	uint64_t num_kvs;

	GPUHashTable(uint64_t cap);

	void insert(const std::vector<KeyValue>& kvs);
	void increment(const std::vector<uint64_t>& keys);
	std::vector<KeyValue> lookup(const std::vector<KeyValue>& kvs);

	std::vector<KeyValue> iterate_hashtable();

	cuda_unique_ptr<KeyValue> get_device_kvs();
};

// murmurhash3 (fmix64)
__device__ uint64_t hash(uint64_t h, uint64_t capacity); // __device__ call by gpu only

__global__ void gpu_hashtable_reset(KeyValue* hashtable, uint64_t capacity);
__global__ void gpu_hashtable_insert(KeyValue* hashtable, uint64_t capacity, KeyValue* kvs, uint64_t num_kvs);
__global__ void gpu_hashtable_lookup(KeyValue* hashtable, uint64_t capacity, KeyValue* kvs, uint64_t num_kvs);
__global__ void gpu_hashtable_incr(KeyValue* hashtable, uint64_t capacity, uint64_t* keys, uint64_t num_keys);

__global__ void gpu_iterate_hashtable(KeyValue* hashtable, uint64_t capacity, KeyValue* kvs, uint64_t* num_kvs);

uint32_t next_pow2(uint32_t bound);
