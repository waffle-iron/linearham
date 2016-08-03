
#include "smooshable_chain.hpp"

/// @file smooshable_stack.hpp
/// @brief Headers for SmooshableStack class.

namespace linearham {


class SmooshableStack {
 protected:
  std::vector<Eigen::MatrixXd> marginals_;
  std::vector<Eigen::MatrixXd> viterbis_;
  std::vector<int> scaler_counts_;
  std::vector<std::string> labels_;
  
 public:
  SmooshableStack(){};
  SmooshableStack(int left_flex, int right_flex, int size);
  SmooshableStack(const SmooshableVector& smooshs);
  
  int size() const { return marginals_.size(); };
  int left_flex() const { return marginals_[0].rows(); };
  int right_flex() const { return marginals_[0].cols(); };
  
  const std::vector<Eigen::MatrixXd> marginals() const { return marginals_; };
  std::vector<Eigen::MatrixXd>& marginals() { return marginals_; };
  
  const std::vector<Eigen::MatrixXd> viterbis() const { return viterbis_; };
  std::vector<Eigen::MatrixXd>& viterbis() { return viterbis_; };
  
  std::vector<int> scaler_counts() const { return scaler_counts_; };
  std::vector<int>& scaler_counts() { return scaler_counts_; };
  
  std::vector<std::string> labels() const { return labels_; };
  std::vector<std::string>& labels() { return labels_; };
};


/// @brief Smoosh two SmooshableStacks!
std::pair<SmooshableStack, std::vector<Eigen::MatrixXi>> SmooshStack(const SmooshableStack& s_a,
                                                                     const SmooshableStack& s_b);


}
