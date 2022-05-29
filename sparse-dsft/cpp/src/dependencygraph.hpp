#ifndef DEPENDENCYGRAPH_HPP
#define DEPENDENCYGRAPH_HPP

#include <cmath>
#include <cstring>
#include <set>
#include "common.hpp"

// TODO Graph structure: is set A < set B? (however we define our order)
class DependencyGraph {

public:

  DependencyGraph(bool* x, c_real* y, int32_t samples, int32_t features) {
    build_graph(x, y, samples, features);
  }
  
  ~DependencyGraph() {
    if (root->idx == -1) {
      delete root;
    }
    for (int32_t i = 0; i < count; ++i) {
      delete all_nodes[i];
    }
    delete[] all_nodes;
  }
  
  c_real compute_max_correlation();
  c_real compute_max_correlation(bool* x, c_real* y, int32_t samples, int32_t features, bool* b, bool* fb);
  void print_dependencies();
  
private:

  class Node {
  public:
    c_real y;
    int32_t idx;
    bool processed;
    std::set<Node*> children;
    Node(int32_t n, c_real yv) : y(yv), idx(n), processed(false) {}
  };
  
  Node* root; // The max element. If non-existent in the data set, row_idx == -1
  Node** all_nodes;
  int32_t count;

  void build_graph(bool* x, c_real* y, int32_t samples, int32_t features);
  bool is_parent(bool* x, int32_t s1, int32_t s2, int32_t features);
  bool is_root(bool* x, int32_t s, int32_t features);
  void remove_redundancy();
  void remove_redundancy_rec(Node* node);
  c_real compute_max_correlation_rec(Node* node);
  void compute_max_correlation_rec(Node* node, bool* x, c_real* y, bool* best_freq,
      bool* v, int32_t top1, int32_t samples, int32_t features, c_real& max_corr);
  void add_children(Node* node, c_real& mu1, c_real& mu2);
  void print_dependencies_rec(Node* node, int32_t indent);
  void clean_node_flags(Node* node);
  
  
};
  
  

#endif
