
#include "smooshable_chain.hpp"

/// @file smooshable_stack.hpp
/// @brief Headers for SmooshableStack class.

namespace linearham {


class SmooshableStack {
 protected:
  SmooshableVector smooshables_;
  
 public:
  SmooshableStack(){};
  SmooshableStack(int left_flex, int right_flex, int size);
  SmooshableStack(const SmooshableVector& smooshables);
  
  int size() const { return smooshables_.size(); };
  int left_flex() const { return smooshables_[0].marginal().rows(); };
  int right_flex() const { return smooshables_[0].marginal().cols(); };
  
  SmooshableVector smooshables() const { return smooshables_; };
  SmooshableVector& smooshables() { return smooshables_; };
};


/// @brief Smoosh two SmooshableStacks!
std::pair<SmooshableStack, std::vector<Eigen::MatrixXi>> Smoosh(const SmooshableStack& s_a,
                                                                const SmooshableStack& s_b);
}
