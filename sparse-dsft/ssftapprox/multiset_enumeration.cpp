#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <queue>
#include <vector>
#include <cstdlib>
#include <iostream>
#include <atomic>
#include <unordered_map>
#include <bitset>
#include <omp.h>


namespace py = pybind11;

using namespace std;

double ms_compute_max_correlation(py::array_t<int> X, 
							 py::array_t<double> Y,
							 py::array_t<int>& B,
							 py::array_t<bool>& fB,
                             py::array_t<int> M) {
	/* 
	n is the size of the ground set
	dimensions:
	X ... n_samples x n 
	Y ... n_samples 
	B ... n multiset
	fB ... n_samples 
	M ... the ground-multiset
	*/
	
	auto x = X.unchecked<2>();
	auto y = Y.unchecked<1>();
	auto b = B.mutable_unchecked<1>();
	auto fb = fB.mutable_unchecked<1>();	
	auto m = M.unchecked<1>(); 
	int n = x.shape(1);
    vector<int> ground_set;
    for (int i = 0; i < n; i++){
        ground_set.push_back(m(i));
    }
	
	vector<int> best_freq(n, 0);// emptyset is 0 0 0 0 0 0 0  vector, entry i has multiplicity of i-th element in the groundset
	double best_corr = numeric_limits<double>::min();
		
	queue<vector<int>> q;
	queue<double> mus;
	for (int i = 0; i < n; i++){
        for (int x = 1; x <= ground_set[i]; x++){
            vector<int> freq(n, 0);
            freq[i] = x;
            q.push(freq);
            mus.push(numeric_limits<double>::max());
        }
	}
	
	while (!q.empty()) {
		auto freq = q.front();
		q.pop();
        double mu_curr = mus.front();
        mus.pop();
        if(mu_curr < abs(best_corr)){
            continue;
        }
		double mu1 = 0;
		double mu2 = 0;
		for (int i = 0; i < x.shape(0); i++) {
			bool is_subset = true;
            for (int j = 0; j < n; j++){
                if (freq[j] > x(i, j)){
                    is_subset = false;
                    break;
                }
            }
			if (is_subset) {
				if (y(i) > 0) {
					mu1 += y(i);
				}
				else {
					mu2 -= y(i);
				}
			}
		}
		double corr = mu1 - mu2;
		if (abs(corr) > abs(best_corr)) {
			best_corr = corr;
			best_freq = freq;
		}
		if (max(mu1, mu2) > abs(best_corr)) {
			int pos = 0;
            for (pos = 0; pos < n; pos++) {
                if (freq[pos] < ground_set[pos]){
                    break;
                }
            }
            for (int i = pos; i < n; i++) {
                vector<int> freq_new(freq);
                freq_new[i] += 1;
                q.push(freq_new);
                mus.push(max(mu1, mu2));
            }
		}
	}
	
	for (size_t i = 0; i < n; i++) {
		b(i) = best_freq[i];
	}
	
	for (size_t i = 0; i < x.shape(0); i++) {
		bool is_subset = true;
		for (int j = 0; j < n; j++){
            if (best_freq[j] > x(i, j)){
                is_subset = false;
                break;
            }
        }
		if (is_subset) {
			fb(i) = 1;
		}
		else {
			fb(i) = 0;
		}
	}
	return best_corr;
}
