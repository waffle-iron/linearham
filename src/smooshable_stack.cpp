
#include "smooshable_stack.hpp"

/// @file smooshable_stack.cpp
/// @brief Implementation of SmooshableStack class.

namespace linearham {

/// @brief "Boring" constructor, which just sets up memory.
SmooshableStack::SmooshableStack(int left_flex, int right_flex, int size) {

  marginals_.resize(size);
  viterbis_.resize(size);
  scaler_counts_.resize(size);
  labels_.resize(size);
  
  for (int i = 0; i < size; i++) {
    marginals_[i].resize(left_flex, right_flex);
    viterbis_[i].resize(left_flex, right_flex);
  }
};


/// @brief Constructor starting from a vector of Smooshables.
SmooshableStack::SmooshableStack(const SmooshableVector& smooshs) {
  
  // checking whether all the 'left_flex' and 'right_flex' values are equal
  for (int i = 0; i < smooshs.size() - 1; i++) {
    assert(smooshs[i].marginal().rows() == smooshs[i+1].marginal().rows());
    assert(smooshs[i].marginal().cols() == smooshs[i+1].marginal().cols());
  }
  
  marginals_.resize(smooshs.size());
  viterbis_.resize(smooshs.size());
  scaler_counts_.resize(smooshs.size());
  labels_.resize(smooshs.size());
  
  for (int i = 0; i < smooshs.size(); i++) {
    marginals_[i] = smooshs[i].marginal();
    viterbis_[i] = smooshs[i].viterbi();
    scaler_counts_[i] = smooshs[i].scaler_count();
    labels_[i] = std::to_string(i);
  }
};


/// @brief Smoosh two SmooshableStacks!
std::pair<SmooshableStack, std::vector<Eigen::MatrixXi>> SmooshStack(const SmooshableStack& s_a,
                                                                     const SmooshableStack& s_b) {
  assert(s_a.right_flex() == s_b.left_flex());
  
  SmooshableStack s_out(s_a.left_flex(), s_b.right_flex(), s_a.size() * s_b.size());
  std::vector<Eigen::MatrixXi> viterbi_idxs(s_a.size() * s_b.size());
  
  for (int i = 0; i < s_a.size(); i++) {
    for (int j = 0; j < s_b.size(); j++) {
      viterbi_idxs[(s_b.size()*i)+j].resize(s_a.left_flex(), s_b.right_flex());
      
      s_out.marginals()[(s_b.size()*i)+j] = s_a.marginals()[i] * s_b.marginals()[j];
      BinaryMax(s_a.viterbis()[i], s_b.viterbis()[j],
                s_out.viterbis()[(s_b.size()*i)+j], viterbi_idxs[(s_b.size()*i)+j]);
      s_out.scaler_counts()[(s_b.size()*i)+j] = s_a.scaler_counts()[i] + s_b.scaler_counts()[j];
      s_out.labels()[(s_b.size()*i)+j] = s_a.labels()[i] + "," + s_b.labels()[j];
    }
  }
  
  return std::make_pair(s_out, viterbi_idxs);
};
}

