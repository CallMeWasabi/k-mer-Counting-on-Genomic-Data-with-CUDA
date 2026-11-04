#include <string>
#include <vector>

// Read DNA stream file and convert to 2-bit base format
class DNALoader {
private:
	std::string dna;
	std::vector<uint64_t> breaks; // position in dna where a new segment start afte N run
	std::vector<uint64_t> words;

public:
	DNALoader(char* filepath);

	std::string& get_dna();
	std::vector<size_t>& get_breaks();
	std::vector<uint64_t>& get_words();
};

uint32_t next_pow2(uint32_t bound);

std::string word_to_str(uint64_t word, uint32_t k);
