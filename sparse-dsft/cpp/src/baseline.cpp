#include "baseline.hpp"

uint64_t bas_it_ctr;

c_real compute_max_correlation_baseline(bool* x, c_real* y, bool* b, bool* fb, int32_t samples, int32_t features) {

	size_t n = features;
	std::vector<int32_t> best_freq;// emptyset is empty std::vector, {0 2 3} is 0 2 3, etc.
	c_real best_corr = 0;
		
	std::queue<std::vector<int32_t>> q;
	//~ q.push(best_freq);
  
  for (size_t i = 0; i < n; i++) {
    std::vector<int32_t> freq_new;
    freq_new.push_back(i);
    q.push(freq_new);
  }
      
	while (!q.empty()) {
     //~ ++bas_it_ctr;
		auto freq = q.front();
		q.pop();
		c_real mu1 = 0;
		c_real mu2 = 0;
		for (size_t i = 0; i < samples; i++) {
			bool is_subset = true;
			for (int32_t k : freq){
				if (!x[i * features + k]) {
					is_subset = false;
					break;
				}
			}
			if (is_subset) {
				if (y[i] > 0) {
					mu1 += y[i];
				}
				else {
					mu2 -= y[i];
				}
			}
		}
    //~ std::cout << "F [ ";
    //~ for (int32_t i = 0; i < freq.size(); ++i) {
      //~ std::cout << freq[i] << " ";
    //~ }
    //~ std::cout << "] " << mu1 << " " << mu2 << " " << best_corr << "\n";
		c_real corr = mu1 - mu2;
    //~ std::cout << "CANDIDATE " << corr << "\n";
		if (std::fabs(corr) > std::fabs(best_corr)) {
			best_corr = corr;
      // best freq is a vector with highest correlation (will be a Node in graph setup)
			best_freq = freq;
		}
		if (std::max(mu1, mu2) > std::fabs(best_corr)) {
      //~ std::cout << "candidate accepted\n";
			int32_t value;
			if (!freq.empty()){
				value = freq.back();
			}
			else {
				value = -1;
			}
			
			for (size_t i = value + 1; i < n; i++) {
				std::vector<int32_t> freq_new(freq);
				freq_new.push_back(i);
				q.push(freq_new);
			}
		} else {
      //~ std::cout << "candidate discarded\n";
    }
	}
  
  for (size_t i = 0; i < n; i++) {
		b[i] = 0;
	}
	
	for (int k : best_freq) {
		b[k] = true;
	}
	for (size_t i = 0; i < samples; i++) {
		bool is_subset = true;
		for (int k : best_freq){
			if (!x[i * n + k]) {
				is_subset = false;
				break;
			}
		}
		if (is_subset) {
			fb[i] = 1;
		}
		else {
			fb[i] = 0;
		}
	}
	return best_corr;
}

c_real compute_max_correlation_recursive_baseline(bool* x, c_real* y, bool* b, bool* fb, int32_t samples, int32_t features) {
	
	/* copy the data int32_to CPP */
	size_t m = samples;
	size_t n = features;
	
	std::vector<int32_t> best_freq;// emptyset is 0 0 0 0, {2 3} is 2 3 0 0 0, etc.
	std::vector<int32_t> freq;
	c_real best_corr = 0;
	std::vector<int32_t> active(m, 0);
	for(int32_t i = 0; i < m; i++){
		active[i] = i;
	}
	traverse_rec(freq, best_corr, best_freq, active, x, y, samples, features);
	
  for (size_t i = 0; i < n; i++) {
		b[i] = 0;
	}
	
	for (int k : best_freq) {
		b[k] = true;
	}
	for (size_t i = 0; i < samples; i++) {
		bool is_subset = true;
		for (int k : best_freq){
			if (!x[i * n + k]) {
				is_subset = false;
				break;
			}
		}
		if (is_subset) {
			fb[i] = 1;
		}
		else {
			fb[i] = 0;
		}
	}
  
	return best_corr;
}

void traverse_rec(std::vector<int32_t>& freq, c_real& best_corr, std::vector<int32_t>& best_freq,
    std::vector<int32_t>& active, bool* x, c_real* y, int32_t samples, int32_t features){
      
      //~ ++bas_it_ctr;
      
	int32_t m = samples;
	c_real mu1 = 0;
	c_real mu2 = 0;
	std::vector<int32_t>  active_new;
	int32_t n = -1;
	for(int32_t i: active) {
		bool is_subset = true;
		n = features;
		for (int32_t k: freq) {
			if (!x[i * features + k]) {
				is_subset = false;
				break;
			}
		}
		if (is_subset) {
			if (y[i] > 0) {
				mu1 += y[i];
			}
			else {
				mu2 -= y[i];
			}
			active_new.push_back(i);
              //~ std::cout << "is in " << i << "\n";
		}
	}
	
	if(n > 0){
		c_real corr = mu1 - mu2;
		if (std::fabs(corr) > std::fabs(best_corr)) {
			best_corr = corr;
			best_freq = freq;
		}
		if (std::max(mu1, mu2) > std::fabs(best_corr)){
			int32_t value;
			if (!freq.empty()) {
				value = freq.back();
			}
			else{
				value = -1;
			}
				
			for (int32_t i = value + 1; i < n; i++) {
				std::vector<int32_t> freq_new(freq);
				freq_new.push_back(i);
        //~ int32_t k = 0;
        //~ int32_t k2 = 0;
        //~ int32_t s = freq_new.size();
        //~ while (k < features) {
          //~ if (k2 < s) {
            //~ int32_t idx = freq_new[k2];
            //~ while (k < idx) {
              //~ std::cout << '0';
              //~ ++k;
            //~ }
            //~ std::cout << '1';
            //~ ++k;
            //~ ++k2;
          //~ } else {
            //~ std::cout << '0';
            //~ ++k;
          //~ }
        //~ }
        //~ std::cout << "( ";
        //~ for (auto kk : freq_new) {
          //~ std::cout << kk << ' ';
        //~ }
        //~ std::cout << ")"
        //~ std::cout << "\n";
				traverse_rec(freq_new, best_corr, best_freq, active_new, x, y, samples, features);
			}
		}
	}
}
