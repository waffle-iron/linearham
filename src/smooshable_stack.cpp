
#include "smooshable_stack.hpp"

/// @file smooshable_stack.cpp
/// @brief Implementation of SmooshableStack class.

namespace linearham {

/// @brief "Boring" constructor, which just sets up memory.
SmooshableStack::SmooshableStack(int left_flex, int right_flex, int n) {

  marginals_.resize(n);
  viterbis_.resize(n);
  labels_.resize(n);
  
  for (int i = 0; i < n; i++) {
    
    marginals_[i].resize(left_flex, right_flex);
    viterbis_[i].resize(left_flex, right_flex);
    
  }
  
};


/// @brief Constructor starting from a vector of Smooshables.
SmooshableStack::SmooshableStack(const SmooshableVector& smooshs) {
  
  // If there's only one Smooshable, there is nothing to stack.
  if (smooshs.size() >= 2) {

    for (int i = 0; i < smooshs.size() - 1; i++) {
  
      assert(smooshs[i].marginal().rows() == smooshs[i+1].marginal().rows());
      assert(smooshs[i].marginal().cols() == smooshs[i+1].marginal().cols());
      assert(smooshs[i].scaler() == smooshs[i+1].scaler());
    
    }
  }
  
  scaler_ = smooshs[0].scaler();
  marginals_.resize(smooshs.size());
  viterbis_.resize(smooshs.size());
  labels_.resize(smooshs.size());
  
  for (int i = 0; i < smooshs.size(); i++) {
    
    marginals_[i] = smooshs[i].marginal();
    viterbis_[i] = smooshs[i].viterbi();
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
      
      s_out.marginal((s_b.size()*i)+j) = s_a.marginal(i) * s_b.marginal(j);
      BinaryMax(s_a.viterbi(i), s_b.viterbi(j),
                s_out.viterbi((s_b.size()*i)+j), viterbi_idxs[(s_b.size()*i)+j]);
      s_out.label((s_b.size()*i)+j) = s_a.label(i) + "," + s_b.label(j);
      
    }
  }
  
  s_out.scaler() = s_a.scaler() * s_b.scaler();
  
  return std::make_pair(s_out, viterbi_idxs);
};

}

