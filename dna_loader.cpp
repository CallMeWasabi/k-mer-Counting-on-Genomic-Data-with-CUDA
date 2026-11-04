#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include "dna_loader.hpp"

DNALoader::DNALoader(char* filepath) {
	std::ifstream file(filepath);
	if (!file.is_open()) {
		std::cerr << "Error cannot open file" << std::endl;
		return;
	}

	std::string buf;
	bool in_n = false;

	file.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); // skip first line
	while (std::getline(file, buf)) {
		std::string remove_n;
		for (char& c : buf) {
			if (c == 'N' || c == 'n') {
				in_n = true;
				continue;
			}
			if (in_n) {
				breaks.push_back(dna.length() + remove_n.length());
				in_n = false;
			}
			remove_n += c;
		}

		dna.append(remove_n);
	}
	file.close();

	size_t arr_cap = (dna.length() + 32 - 1) / 32;
	words.resize(arr_cap);

	static char lookup[128] = {0};
	lookup['A'] = 0;
	lookup['C'] = 1;
	lookup['G'] = 2;
	lookup['T'] = 3;

	for (int32_t i = 0, j = 0; i < dna.length(); i++) {
		if (i % 32 == 0 && i != 0) j++;

		char c = dna[i];
		uint8_t val = lookup[c];
		words[j] = (words[j] << 2) | val;
	}

	// left-align the last partial word so its first base is in the top bits
	if (dna.length() % 32 != 0) words[arr_cap - 1] <<= 2 * (32 - dna.length() % 32);
}

std::string& DNALoader::get_dna() { return dna; }

std::vector<size_t>& DNALoader::get_breaks() { return breaks; }

std::vector<uint64_t>& DNALoader::get_words() { return words; }

// convert k-mer base in 2-bit format to string
std::string word_to_str(uint64_t word, uint32_t k) {
	static const char c_lookup[4] = {'A', 'C', 'G', 'T'};
	std::string result;
	for (size_t i = 0; i < k; i++) {
		uint64_t bits = (word >> (2 * (k - 1 - i))) & 3;
		char c = c_lookup[bits];
		result += c;
	}
	return result;
}
