#include "dependencygraph.hpp"

// TODO all nodes should be bigger than their children
// if the full set exists in the dataset, set root idx to it, otherwise root idx == -1
void DependencyGraph::build_graph(bool* x, c_real* y, int32_t samples, int32_t features) {
  
  count = samples;
  all_nodes = new Node*[samples]; // a temp data structure for storing and connecting nodes
  root = new Node(-1, 0.0);
  for (int32_t s1 = 0; s1 < samples; ++s1) {
    // Assuming max one root in the dataset
    if (is_root(x, s1, features)) {
      root->idx = s1;
      root->y = y[s1];
      all_nodes[s1] = root;
    } else {
      Node* node1 = new Node(s1, y[s1]);
      root->children.insert(node1);
      all_nodes[s1] = node1;
      for (int32_t s2 = 0; s2 < s1; ++s2) {
        if (root->idx != s2) {
          if (is_parent(x, s1, s2, features)) {
            all_nodes[s1]->children.insert(all_nodes[s2]);
          } else if (is_parent(x, s2, s1, features)) {
            all_nodes[s2]->children.insert(all_nodes[s1]);
          }
        }
      }
    }
  }
  remove_redundancy();
}

bool DependencyGraph::is_parent(bool* x, int32_t s1, int32_t s2, int32_t features) {
  for (int32_t f = 0; f < features; ++f) {
    if (x[s1 * features + f] > x[s2 * features + f]) {
      return false;
    }
  }
  return true;
}

bool DependencyGraph::is_root(bool* x, int32_t s, int32_t features) {
  for (int32_t f = 0; f < features; ++f) {
    if (x[s * features + f]) {
      return false;
    }
  }
  return true;
}

void DependencyGraph::remove_redundancy() {
  remove_redundancy_rec(root);
  clean_node_flags(root);
}

void DependencyGraph::remove_redundancy_rec(Node* node) {
  if (!node->processed) {
    node->processed = true;
    for (Node* child : node->children) {
      for (Node* grandchild : child->children) {
        // remove items from set while iterating over it
        for (auto sibling = node->children.begin(); sibling != node->children.end();) {
          if (*sibling == grandchild) {
           node->children.erase(sibling++);
          } else {
            ++sibling;
          }
        }
      }
      remove_redundancy_rec(child);
    }
  }
}

c_real DependencyGraph::compute_max_correlation() {
  c_real max_corr = compute_max_correlation_rec(root);
  clean_node_flags(root);
  return max_corr;
}

c_real DependencyGraph::compute_max_correlation_rec(Node* node) {
  c_real max_corr = 0.0;
  //~ if (!node->processed) {
    //~ node->processed = true;
    c_real mu1 = 0.0;
    c_real mu2 = 0.0;
    for (Node* child : node->children) {
      c_real y = child->y;
      if (y > 0) {
        mu1 += y;
      } else {
        mu2 -= y;
      }
    }
    c_real new_corr = mu1 - mu2;
    if (std::fabs(new_corr) > std::fabs(max_corr)) {
      max_corr = new_corr;
    }
    if (std::max(mu1, mu2) >= std::fabs(max_corr)) {
      for (Node* child : node->children) {
        c_real rec_corr = compute_max_correlation_rec(child);
        if (std::fabs(rec_corr) > std::fabs(max_corr)) {
          max_corr = rec_corr;
        }
      }
    }
  //~ }
  return max_corr;
}

c_real DependencyGraph::compute_max_correlation(bool* x, c_real* y, int32_t samples, int32_t features,
    bool* b, bool* fb) {
  bool* v = new bool[features];
  std::memset(v, 0, sizeof(bool) * features);
  int32_t top1 = -1;
  c_real max_corr = 0.0;
  compute_max_correlation_rec(root, x, y, b, v, top1, samples, features, max_corr);

	for (size_t i = 0; i < samples; i++) {
		bool is_subset = true;
		for (size_t j = 0; j < features; ++j){
			if (b[j] && !x[i * features + j]) {
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
  
  delete[] v;
  return max_corr;
}

void DependencyGraph::compute_max_correlation_rec(Node* node, bool* x, c_real* y, bool* best_freq,
    bool* v, int32_t top1, int32_t samples, int32_t features, c_real& max_corr) {
  c_real mu1 = 0.0;
  c_real mu2 = 0.0;

  //~ for (int32_t i = 0; i < count; ++i) {
    //~ if (!all_nodes[i]->processed) {
      //~ int32_t j = 0;
      //~ for (; j < features; ++j) {
        //~ if (x[i * features + j] < v[j]) {
          //~ break;
        //~ }
      //~ }
      //~ if (j == features) {
        //~ add_children(all_nodes[i], mu1, mu2);
      //~ }
    //~ }
  //~ }
  //~ c_real new_corr = mu1 - mu2;
  //~ for (int32_t i = 0; i < count; ++i) {
    //~ all_nodes[i]->processed = false;
  //~ }
  
  for (int32_t i = 0; i < count; ++i) {
    int32_t j = 0;
    for (; j < features; ++j) {
      if (x[i * features + j] < v[j]) {
        break;
      }
    }
    if (j == features) {
      c_real y1 = y[i];
      if (y1 > 0) {
        mu1 += y1;
      } else {
        mu2 -= y1;
      }
    }
  }
  c_real new_corr = mu1 - mu2;
  for (int32_t i = 0; i < count; ++i) {
    all_nodes[i]->processed = false;
  }
  if (std::fabs(new_corr) > std::fabs(max_corr)) {
    max_corr = new_corr;
    std::memcpy(best_freq, v, features * sizeof(bool));
  }
  //~ std::cout << "<";
  //~ for (int32_t j = 0; j < features; ++j) {
    //~ std::cout << v[j];
  //~ }
  //~ std::cout << "> " << mu1 << " " << mu2 << " " << max_corr << "\n";
  if (std::max(mu1, mu2) >= std::fabs(max_corr)) {
    for (int32_t nt = top1 + 1; nt < features; ++nt) {
      v[nt] = true;
      compute_max_correlation_rec(node, x, y, best_freq, v, nt, samples, features, max_corr);
      v[nt] = false;
    }
  }
}

void DependencyGraph::add_children(Node* node, c_real& mu1, c_real& mu2) {
  if (!node->processed) {
    node->processed = true;
    c_real y = node-> y;
    if (y > 0) {
      mu1 += y;
    } else {
      mu2 -= y;
    }
    for (Node* child : node->children) {
      add_children(child, mu1, mu2);
    }
  }
}

void DependencyGraph::print_dependencies() {
  print_dependencies_rec(root, 0);
  clean_node_flags(root);
  std::cout << "Iteratively:\n";
  for (int32_t i = 0; i < count; ++i) {
    std::cout << all_nodes[i]->idx << " " << all_nodes[i]->children.size() << "\n";
  }
}

void DependencyGraph::print_dependencies_rec(Node* node, int32_t indent) {
  for (int32_t i = 0; i < indent; ++i) {
    std::cout << "  ";
  }
  std::cout << node->idx << "\n";
  for (Node* child : node->children) {
    print_dependencies_rec(child, indent + 1);
  }
}

void DependencyGraph::clean_node_flags(Node* node) {
  if (node->processed) {
    node->processed = false;
    for (Node* child : node->children) {
      clean_node_flags(child);
    }
  }
}
