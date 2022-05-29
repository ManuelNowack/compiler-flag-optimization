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

double sgn(double val) {
	if (val <  0)
		return -1.0;
	else
		return 1.0;
}


double maximize_fourier_sparse(py::array_t<bool> X, 
							 py::array_t<double> Y,
							 py::array_t<bool>& B,
                             bool headstart) {
	/* 
	n is the size of the ground set
	dimensions:
	X ... n_samples x n frequiencies
	Y ... n_samples coefficients
	B ... n argmax
	fB ... n_samples 
	*/
	
	auto x = X.unchecked<2>();
	auto y = Y.unchecked<1>();
	auto b = B.mutable_unchecked<1>();
	
	
	int n = x.shape(1);
    int k = x.shape(0);
	vector<int> best_freq;// empty vector == N, 0 = N \ {x_n} and so on
	double best_corr = numeric_limits<double>::min();
	
    queue<vector<int>> q;
    queue<double> mus;
    
	if (headstart) {
        for (int i = 0; i < k; i++){
            if(y(i) >= 0) {
                vector<int> freq;
                for(int j = 0; j < n; j++){
                    if (x(i, j)) {
                        freq.push_back(j);
                    }
                }
                q.push(freq);
                mus.push(numeric_limits<double>::max());
            }
        }
    }
    q.push(best_freq);
    mus.push(numeric_limits<double>::max());

    
    
    
    
    
	while (!q.empty()) {
		auto freq = q.front();
		q.pop();
        double mu_curr = mus.front();
        mus.pop();
        if(mu_curr < best_corr){
            continue;
        }
		double mu1 = 0;
		double mu2 = 0;
		for (size_t i = 0; i < x.shape(0); i++) {
			bool is_subset = true;
			for (int k : freq){
				if (x(i, n - 1 - k)) {
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
		if (corr > best_corr) {
			best_corr = corr;
			best_freq = freq;
		}
		if (mu1 > best_corr) {
			int value;
			if (!freq.empty()){
				value = freq.back();
			}
			else {
				value = -1;
			}
			
			for (size_t i = value + 1; i < n; i++) {
				vector<int> freq_new(freq);
				freq_new.push_back(i);
				q.push(freq_new);
                mus.push(mu1);
			}
		}
	}
	
	for (size_t i = 0; i < n; i++) {
		b(i) = 1;
	}
	
	for (int k : best_freq) {
		b(n - 1 - k) = 0;
	}

	return best_corr;
}

double compute_max_positive_correlation(py::array_t<bool> X, 
							 py::array_t<double> Y,
							 py::array_t<bool>& B,
							 py::array_t<bool>& fB,
                             int32_t max_cardinality) {
	/* 
	n is the size of the ground set
	dimensions:
	X ... n_samples x n 
	Y ... n_samples 
	B ... n 
	fB ... n_samples 
	*/
	
	auto x = X.unchecked<2>();
	auto y = Y.unchecked<1>();
	auto b = B.mutable_unchecked<1>();
	auto fb = fB.mutable_unchecked<1>();	
	
	
	size_t n = x.shape(1);
	vector<int> best_freq;// emptyset is empty vector, {0 2 3} is 0 2 3, etc.
	double best_corr = numeric_limits<double>::min();
		
	queue<vector<int>> q;
	q.push(best_freq);
    queue<double> mus;
	mus.push(1);

	
	while (!q.empty()) {
		auto freq = q.front();
		q.pop();
        double mu_curr = mus.front();
        mus.pop();
        if(mu_curr < best_corr) {
            continue;
        }
		double mu1 = 0;
		double mu2 = 0;
		for (size_t i = 0; i < x.shape(0); i++) {
			bool is_subset = true;
			for (int k : freq){
				if (!x(i, k)) {
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
		if (corr > best_corr) {
			best_corr = corr;
			best_freq = freq;
		}
		if (mu1 > best_corr && freq.size() < max_cardinality) {
			int value;
			if (!freq.empty()){
				value = freq.back();
			}
			else {
				value = -1;
			}
			
			for (size_t i = value + 1; i < n; i++) {
				vector<int> freq_new(freq);
				freq_new.push_back(i);
				q.push(freq_new);
                mus.push(mu1);
			}
		}
	}
	
	for (size_t i = 0; i < n; i++) {
		b(i) = 0;
	}
	
	for (int k : best_freq) {
		b(k) = true;
	}
	for (size_t i = 0; i < x.shape(0); i++) {
		bool is_subset = true;
		for (int k : best_freq){
			if (!x(i, k)) {
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

void traverse_rec_positive(vector<int> & freq, double & best_corr, vector<int> & best_freq, vector<int>& active, vector<vector<bool>>& x, vector<double>& y){
	int m = y.size();
	double mu1 = 0;
	double mu2 = 0;
	vector<int>  active_new;
	int n = x[0].size();
	for(int i: active) {
		bool is_subset = true;
		for (int k: freq) {
			if (!x[i][k]) {
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
		}
	}
	
	double corr = mu1 - mu2;
	if (corr > best_corr) {
		best_corr = corr;
		best_freq = freq;
	}
	if (mu1 >= best_corr){
		int value;
		if (!freq.empty()) {
			value = freq.back();
		}
		else{
			value = -1;
		}
			
		for (size_t i = value + 1; i < n; i++) {
			vector<int> freq_new(freq);
			freq_new.push_back(i);
			traverse_rec_positive(freq_new, best_corr, best_freq, active_new, x, y);
		}
	}
	if (corr > best_corr) {
		best_corr = corr;
		best_freq = freq;
	}
	
}

double compute_max_positive_correlation_recursive(py::array_t<bool> X, 
							 py::array_t<double> Y,
							 py::array_t<bool>& B,
							 py::array_t<bool>& fB) {
	/* 
	n is the size of the ground set
	dimensions:
	X ... n_samples x n 
	Y ... n_samples 
	B ... n 
	fB ... n_samples 
	*/
	
	auto x_np = X.unchecked<2>();
	auto y_np = Y.unchecked<1>();
	
	/* copy the data into CPP */
	size_t m = x_np.shape(0);
	size_t n = x_np.shape(1);
	vector<vector<bool>> x(m, vector<bool>(n));
	vector<double> y(m);
	
	for(size_t i = 0; i < m; i++) {
		y[i] = y_np(i);
		for(size_t j = 0; j < n; j++) {
			x[i][j] = x_np(i, j);
		}
	}
	
	vector<int> best_freq;
	vector<int> freq;
	double best_corr = 0;
	vector<int> active(m, 0);
	for(int i = 0; i < m; i++){
		active[i] = i;
	}
	
	traverse_rec_positive(freq, best_corr, best_freq, active, x, y);
	
	
	auto b = B.mutable_unchecked<1>();
	auto fb = fB.mutable_unchecked<1>();	
	for (size_t i = 0; i < n; i++) {
		b(i) = 0;
	}
	
	for (int k: best_freq){
		b(k) = true;
	}
	

	for (size_t i = 0; i < m; i++) {
		bool is_subset = true;
		for (int k: best_freq) {
			if (!x[i][k]) {
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


double compute_max_correlation(py::array_t<bool> X, 
							 py::array_t<double> Y,
							 py::array_t<bool>& B,
							 py::array_t<bool>& fB) {
	/* 
	n is the size of the ground set
	dimensions:
	X ... n_samples x n 
	Y ... n_samples 
	B ... n 
	fB ... n_samples 
	*/
	
	auto x = X.unchecked<2>();
	auto y = Y.unchecked<1>();
	auto b = B.mutable_unchecked<1>();
	auto fb = fB.mutable_unchecked<1>();	
	
	
	size_t n = x.shape(1);
	vector<int> best_freq;// emptyset is empty vector, {0 2 3} is 0 2 3, etc.
	double best_corr = 0;
		
	queue<vector<int>> q;
	queue<double> mus;
    //q.push(best_freq);
	for (int i = 0; i < n; i++){
		vector<int> freq;
		freq.push_back(i);
		q.push(freq);
        mus.push(numeric_limits<double>::max());
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
		for (size_t i = 0; i < x.shape(0); i++) {
			bool is_subset = true;
			for (int k : freq){
				if (!x(i, k)) {
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
			int value;
			if (!freq.empty()){
				value = freq.back();
			}
			else {
				value = -1;
			}
			
			for (size_t i = value + 1; i < n; i++) {
				vector<int> freq_new(freq);
				freq_new.push_back(i);
				q.push(freq_new);
                mus.push(max(mu1, mu2));
			}
		}
	}
	
	for (size_t i = 0; i < n; i++) {
		b(i) = 0;
	}
	
	for (int k : best_freq) {
		b(k) = true;
	}
	for (size_t i = 0; i < x.shape(0); i++) {
		bool is_subset = true;
		for (int k : best_freq){
			if (!x(i, k)) {
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


double compute_max_correlation_parallel(py::array_t<bool> X, 
							 py::array_t<double> Y,
							 py::array_t<bool>& B,
							 py::array_t<bool>& fB) {
	/* 
	n is the size of the ground set
	dimensions:
	X ... n_samples x n 
	Y ... n_samples 
	B ... n 
	fB ... n_samples 
	*/
	
	auto x_np = X.unchecked<2>();
	auto y_np = Y.unchecked<1>();
	
	/* copy the data into CPP */
	size_t m = x_np.shape(0);
	size_t n = x_np.shape(1);
	vector<vector<bool>> x(m, vector<bool>(n));
	vector<double> y(m);
	
	for(size_t i = 0; i < m; i++) {
		y[i] = y_np(i);
		for(size_t j = 0; j < n; j++) {
			x[i][j] = x_np(i, j);
		}
	}
	
	/* do the parallel computing */
	
	py::gil_scoped_release release;
	
	vector<int> best_freq;
	double best_corr = 0;
	
	queue<vector<int>> q;
	//q.push(best_freq);
	for (int i = 0; i < n; i++){
		vector<int> freq;
		freq.push_back(i);
		q.push(freq);
	}
	
	while (!q.empty()) {
		auto freq = q.front();
		q.pop();

		double mu1 = 0;
		double mu2 = 0;
		
		
		#pragma omp parallel for reduction(+: mu1) reduction(-:mu2) num_threads(4)
		for (int i = 0; i < m; i++) {
			bool is_subset = true;
			for (int k: freq) {
				if (!x[i][k]) {
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
		
		double corr = mu1 - mu2;
		if (abs(corr) > abs(best_corr)) {
			best_corr = corr;
			best_freq = freq;
		}
		if (max(mu1, mu2) >= abs(best_corr)){
			int value;
			if(!freq.empty()){
				value = freq.back();
			}
			else{
				value = -1;
			}
			
			for (int i = value + 1; i < n; i++) {
				vector<int> freq_new(freq);
				freq_new.push_back(i);
				q.push(freq_new);
			}
		}
	}
	
	
	py::gil_scoped_acquire acquire;
	
	/* write back the result into python */
	
	auto b = B.mutable_unchecked<1>();
	auto fb = fB.mutable_unchecked<1>();	
	for (size_t i = 0; i < n; i++) {
		b(i) = 0;
	}
	
	for (int k : best_freq) {
		b(k) = true;
	}
	for (size_t i = 0; i < m; i++) {
		bool is_subset = true;
		for (int k : best_freq){
			if (!x[i][k]) {
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

//https://docs.microsoft.com/en-us/cpp/parallel/concrt/reference/concurrent-vector-class?view=msvc-160 maybe try this 
void traverse_rec(vector<int> & freq, double & best_corr, vector<int> & best_freq, vector<int>& active, vector<vector<bool>>& x, vector<double>& y){
	int m = y.size();
	double mu1 = 0;
	double mu2 = 0;
	vector<int>  active_new;
	int n = -1;
	for(int i: active) {
		bool is_subset = true;
		n = x[i].size();
		for (int k: freq) {
			if (!x[i][k]) {
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
		}
	}
	
	if(n > 0){
		double corr = mu1 - mu2;
		if (abs(corr) > abs(best_corr)) {
			best_corr = corr;
			best_freq = freq;
		}
		if (max(mu1, mu2) >= abs(best_corr)){
			int value;
			if (!freq.empty()) {
				value = freq.back();
			}
			else{
				value = -1;
			}
				
			for (int i = value + 1; i < n; i++) {
				vector<int> freq_new(freq);
				freq_new.push_back(i);
				traverse_rec(freq_new, best_corr, best_freq, active_new, x, y);
			}
		}
	}
}

double compute_max_correlation_recursive(py::array_t<bool> X, 
							 py::array_t<double> Y,
							 py::array_t<bool>& B,
							 py::array_t<bool>& fB) {
	/* 
	n is the size of the ground set
	dimensions:
	X ... n_samples x n 
	Y ... n_samples 
	B ... n 
	fB ... n_samples 
	*/
	
	auto x_np = X.unchecked<2>();
	auto y_np = Y.unchecked<1>();
	
	/* copy the data into CPP */
	size_t m = x_np.shape(0);
	size_t n = x_np.shape(1);
	vector<vector<bool>> x(m, vector<bool>(n));
	vector<double> y(m);
	
	for(size_t i = 0; i < m; i++) {
		y[i] = y_np(i);
		for(size_t j = 0; j < n; j++) {
			x[i][j] = x_np(i, j);
		}
	}
	
	vector<int> best_freq;// emptyset is 0 0 0 0, {2 3} is 2 3 0 0 0, etc.
	vector<int> freq;
	double best_corr = 0;
	vector<int> active(m, 0);
	for(int i = 0; i < m; i++){
		active[i] = i;
	}
	//traverse_rec(freq, best_corr, best_freq, active, x, y);
	for (int i = 0; i < n; i++){
		vector<int> freq;
		freq.push_back(i);
		traverse_rec(freq, best_corr, best_freq, active, x, y);
	}
	
	
	
	
	auto b = B.mutable_unchecked<1>();
	auto fb = fB.mutable_unchecked<1>();	
	for (size_t i = 0; i < n; i++) {
		b(i) = 0;
	}
	
	for (int k: best_freq){
		b(k) = true;
	}
	

	for (size_t i = 0; i < m; i++) {
		bool is_subset = true;
		for (int k: best_freq) {
			if (!x[i][k]) {
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
