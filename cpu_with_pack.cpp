#include <chrono>
#include <cstdint>
#include <iostream>
#include <ratio>
#include <string>
#include <unordered_map>
#include <vector>
#include "dna_loader.hpp"

void exec_kmer_hashmap(const std::vector<uint64_t>& words, const size_t n, const std::vector<std::size_t>& breaks,
                       const uint32_t k) {
	uint32_t cap = next_pow2(2 * n);

	std::unordered_map<uint64_t, uint64_t> mp;
	mp.reserve(cap);

	auto start = std::chrono::high_resolution_clock::now();

	// create mask
	uint64_t mask = 0;
	for (int i = 0; i < 2 * k; i++) mask = (mask << 1) | 1;

	uint64_t key = 0;
	uint32_t run = 0; // number of consecutive bases since the last N gap
	size_t b = 0;     // next entry in breaks
	for (size_t i = 0; i < n; i++) {
		// a k-mer must not span an N gap: restart the window
		if (b < breaks.size() && breaks[b] == i) {
			key = 0;
			run = 0;
			b++;
		}

		size_t word_idx = i / 32;
		size_t s = i & 31; // & 31 = % 32
		size_t bits = (words[word_idx] >> (62 - s * 2)) & 3;
		key = ((key << 2) | bits) & mask; // keep only the last k bases
		if (++run >= k) mp[key]++;
	}

	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> duration_ms = end - start;

	std::cout << "k-mer (hashmap, with-pack) Execution time: " << duration_ms.count() << std::endl;

	for (auto& [key, val] : mp) std::cout << word_to_str(key, k) << ": " << val << std::endl;
}

int main(int argc, char* argv[]) {
	if (argc < 3) {
		std::cerr << "Error require filepath and k-value" << std::endl;
		return 1;
	}

	char* filepath = argv[1];
	uint32_t k = atoi(argv[2]);
	DNALoader dna = DNALoader(filepath);

	exec_kmer_hashmap(dna.get_words(), dna.get_dna().length(), dna.get_breaks(), k);

	return 0;
}

/*
    g++ cpu_with_pack.cpp && ./a.exe ./data/ecoli_k12_mg1655.fa
    k-mer (hashmap, with-pack) Execution time: 114.412
*/
