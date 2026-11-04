#include <cuda_runtime_api.h>
#include "gpu_hash_table.hpp"

// every slot starts as key = EMPTY, val = 0
// (cudaMemset writes one byte value everywhere, so it cannot give key and val different values)
__global__ void gpu_hashtable_reset(KeyValue* hashtable, uint64_t capacity) {
	uint64_t thread_id = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
	if (thread_id < capacity) {
		hashtable[thread_id].key = GPU_HASH_EMPTY;
		hashtable[thread_id].val = 0;
	}
}

GPUHashTable::GPUHashTable(uint64_t cap) {
	capacity = cap;
	num_kvs = cap / 2; // max load-factor = 0.5
	device_kvs = make_cuda_unique<KeyValue>(capacity);

	gpu_hashtable_reset<<<(capacity + 255) / 256, 256>>>(device_kvs.get(), capacity);
}

__device__ uint64_t hash(uint64_t h, uint64_t capacity) {
	h ^= h >> 33;
	h *= 0xff51afd7ed558ccd;
	h ^= h >> 33;
	h *= 0xc4ceb9fe1a85ec53;
	h ^= h >> 33;
	return h & (capacity - 1);
}

__global__ void gpu_hashtable_insert(KeyValue* hashtable, uint64_t capacity, KeyValue* kvs, size_t num_kvs) {
	uint32_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
	if (thread_id < num_kvs) {
		uint64_t key = kvs[thread_id].key;
		uint64_t val = kvs[thread_id].val;
		uint64_t slot = hash(key, capacity);

		while (true) {
			uint64_t prev = atomicCAS(&hashtable[slot].key, GPU_HASH_EMPTY, key);
			if (prev == GPU_HASH_EMPTY || prev == key) {
				hashtable[slot].val = val;
				return;
			}

			slot = (slot + 1) & (capacity - 1); // & faster than %
		}
	}
}

void GPUHashTable::insert(const std::vector<KeyValue>& kvs) {
	size_t num_kvs = kvs.size();

	// copy kvs to GPU
	auto _device_input_kvs = make_cuda_unique<KeyValue>(num_kvs);
	cudaMemcpy(_device_input_kvs.get(), kvs.data(), sizeof(KeyValue) * num_kvs, cudaMemcpyHostToDevice);

	int32_t min_grid_size;
	int32_t thread_block_size;

	cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &thread_block_size, gpu_hashtable_insert, 0,
	                                   0); // auto calculate thread block size

	int32_t grid_size = ((size_t)num_kvs + thread_block_size - 1) / thread_block_size;

	gpu_hashtable_insert<<<grid_size, thread_block_size>>>(device_kvs.get(), capacity, _device_input_kvs.get(),
	                                                       num_kvs);
}

__global__ void gpu_hashtable_lookup(KeyValue* hashtable, uint64_t capacity, KeyValue* kvs, uint64_t num_kvs) {
	uint32_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
	if (thread_id < num_kvs) {
		uint64_t key = kvs[thread_id].key;
		uint64_t slot = hash(key, capacity);

		while (true) {
			if (hashtable[slot].key == key) {
				kvs[thread_id].val = hashtable[slot].val;
				return;
			}
			if (hashtable[slot].key == GPU_HASH_EMPTY) {
				kvs[thread_id].val = GPU_HASH_EMPTY;
				return;
			}
			slot = (slot + 1) & (capacity - 1);
		}
	}
}

std::vector<KeyValue> GPUHashTable::lookup(const std::vector<KeyValue>& kvs) {
	size_t num_kvs = kvs.size();

	// copy kvs to GPU
	auto _device_input_kvs = make_cuda_unique<KeyValue>(num_kvs);
	cudaMemcpy(_device_input_kvs.get(), kvs.data(), sizeof(KeyValue) * num_kvs, cudaMemcpyHostToDevice);

	int32_t min_grid_size;
	int32_t thread_block_size;
	cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &thread_block_size, gpu_hashtable_lookup, 0,
	                                   0); // auto calculate thread block size

	int32_t grid_size = ((size_t)num_kvs + thread_block_size - 1) / thread_block_size;

	gpu_hashtable_lookup<<<grid_size, thread_block_size>>>(device_kvs.get(), capacity, _device_input_kvs.get(),
	                                                       num_kvs);

	std::vector<KeyValue> result(num_kvs);
	cudaMemcpy(result.data(), _device_input_kvs.get(), sizeof(KeyValue) * num_kvs, cudaMemcpyDeviceToHost);

	return result;
}

__global__ void gpu_hashtable_incr(KeyValue* hashtable, uint64_t capacity, uint64_t* keys, uint64_t num_keys) {
	uint32_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
	if (thread_id < num_keys) {
		uint64_t key = keys[thread_id];
		uint64_t slot = hash(key, capacity);

		while (true) {
			uint64_t cur = keys[slot];
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

			slot = (slot + 1) & (capacity - 1); // & faster than %
		}
	}
}

void GPUHashTable::increment(const std::vector<uint64_t>& keys) {
	size_t num_keys = keys.size();

	auto device_keys = make_cuda_unique<uint64_t>(num_keys);
	cudaMemcpy(device_keys.get(), keys.data(), sizeof(KeyValue) * num_keys, cudaMemcpyHostToDevice);

	int32_t min_grid_size;
	int32_t thread_block_size;
	cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &thread_block_size, gpu_hashtable_incr, 0,
	                                   0); // auto calculate thread block size

	int32_t grid_size = ((size_t)num_kvs + thread_block_size - 1) / thread_block_size;

	gpu_hashtable_incr<<<grid_size, thread_block_size>>>(device_kvs.get(), capacity, device_keys.get(), num_keys);
}

__global__ void gpu_iterate_hashtable(KeyValue* hashtable, uint64_t capacity, KeyValue* kvs, uint64_t* kvs_size) {
	uint32_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
	if (thread_id < capacity) {
		if (hashtable[thread_id].key != GPU_HASH_EMPTY) {
			uint64_t val = hashtable[thread_id].val;
			if (val != GPU_HASH_EMPTY) {
				size_t size = atomicAdd(kvs_size, 1);
				kvs[size] = hashtable[thread_id];
			}
		}
	}
}

std::vector<KeyValue> GPUHashTable::iterate_hashtable() {
	auto device_num_kvs = make_cuda_unique<size_t>(1);
	cudaMemset(device_num_kvs.get(), 0, sizeof(size_t));

	auto _device_kvs =
	    make_cuda_unique<KeyValue>(num_kvs); // at most num_kvs entries (load <= 0.5); capacity would double GPU memory

	int32_t min_grid_size;
	int32_t thread_block_size;
	cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &thread_block_size, gpu_iterate_hashtable, 0,
	                                   0); // auto calculate thread block size

	int32_t grid_size = ((size_t)capacity + thread_block_size - 1) / thread_block_size;

	gpu_iterate_hashtable<<<grid_size, thread_block_size>>>(device_kvs.get(), capacity, _device_kvs.get(),
	                                                        device_num_kvs.get());

	size_t num_kvs;
	cudaMemcpy(&num_kvs, device_num_kvs.get(), sizeof(size_t), cudaMemcpyDeviceToHost);

	std::vector<KeyValue> kvs(num_kvs);
	cudaMemcpy(kvs.data(), _device_kvs.get(), sizeof(KeyValue) * num_kvs, cudaMemcpyDeviceToHost);

	return kvs;
}

uint32_t next_pow2(uint32_t bound) { return ceil(log2(bound)); }
