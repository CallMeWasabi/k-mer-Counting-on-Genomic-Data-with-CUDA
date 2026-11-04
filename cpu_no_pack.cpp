#include <iostream>
#include <string>
#include <cstdint>
#include <unordered_map>
#include <chrono>
#include "dna_loader.hpp"

void exec_kmer_hashmap(std::string& dna, uint32_t k) {
	std::unordered_map<std::string, uint32_t> mp;

	auto start = std::chrono::high_resolution_clock::now();

	for (int32_t i = 0; i < dna.length() - k + 1; i++) {
		std::string key = dna.substr(i, k);
		int32_t idx = -1;
		for (int32_t j = 0; j < key.length(); j++) {
			if (key[j] == 'N') {
				idx = j;
				break;
			}
		}
		if (idx != -1) {
			i = idx;
			continue;
		}

		mp[key]++;
	}

	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> duration_ms = end - start;

	std::cout << "k-mer (hashmap, no-pack) Execution time: " << duration_ms.count() << std::endl;

	// for (auto &[key, val]: mp) std::cout << key << ": " << val << std::endl;
}

int main(int argc, char* argv[]) {
	if (argc < 3) {
		std::cerr << "Error require filepath and k-value" << std::endl;
		return 1;
	}

	char* filepath = argv[1];
	uint32_t k = atoi(argv[2]);
	DNALoader dna = DNALoader(filepath);

	exec_kmer_hashmap(dna.get_dna(), k);
}
/*
     g++ cpu.cpp && ./a.exe ./data/ecoli_k12_mg1655.fa
     k-mer (hashmap) Execution time: 2124.57
     k-mer (map) Execution time: 6594.06

     g++ cpu.cpp && ./a.exe ./data/grch38_chr1.fa
     k-mer (hashmap) Execution time: 178223
     k-mer (map) Execution time: CRASH (Peak memory use 22Gb and NO-OUTPUT)

 */
