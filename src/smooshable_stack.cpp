
#include "smooshable_stack.hpp"

/// @file smooshable_stack.cpp
/// @brief Implementation of SmooshableStack class.

namespace linearham {

/// @brief "Boring" constructor, which just sets up memory.
SmooshableStack::SmooshableStack(int left_flex, int right_flex, int size) {
  smooshables_.resize(size);
  
  for (int i = 0; i < size; i++) {
    smooshables_[i] = Smooshable(left_flex, right_flex);
  }
};


/// @brief Constructor starting from a vector of Smooshables.
SmooshableStack::SmooshableStack(const SmooshableVector& smooshables) {
  
  // checking whether all the 'left_flex' and 'right_flex' values are equal
  for (int i = 0; i < smooshables.size() - 1; i++) {
    assert(smooshables[i].marginal().rows() == smooshables[i+1].marginal().rows());
    assert(smooshables[i].marginal().cols() == smooshables[i+1].marginal().cols());
  }
  
  smooshables_ = smooshables;
};


/// @brief Smoosh two SmooshableStacks!
std::pair<SmooshableStack, std::vector<Eigen::MatrixXi>> Smoosh(const SmooshableStack& s_a,
                                                                const SmooshableStack& s_b) {
  assert(s_a.right_flex() == s_b.left_flex());
  
  SmooshableStack s_out(s_a.left_flex(), s_b.right_flex(), s_a.size() * s_b.size());
  std::vector<Eigen::MatrixXi> viterbi_idxs(s_a.size() * s_b.size());
  
  for (int i = 0; i < s_a.size(); i++) {
    for (int j = 0; j < s_b.size(); j++) {
      std::tie(s_out.smooshables()[(s_b.size()*i)+j],
               viterbi_idxs[(s_b.size()*i)+j]) = Smoosh(s_a.smooshables()[i], s_b.smooshables()[j]);
    }
  }
  
  return std::make_pair(s_out, viterbi_idxs);
};
}

