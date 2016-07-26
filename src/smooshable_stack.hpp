
#include "smooshable_chain.hpp"

/// @file smooshable_stack.hpp
/// @brief Headers for SmooshableStack class.

namespace linearham {


class SmooshableStack {
  
  protected:
    std::vector<Eigen::MatrixXd> marginals_;
    std::vector<Eigen::MatrixXd> viterbis_;
    double scaler_;
    std::vector<std::string> labels_;
  
  public:
    SmooshableStack(){};
    SmooshableStack(int left_flex, int right_flex, int n);
    SmooshableStack(const SmooshableVector& smooshs);
    
    int size() const { return labels_.size(); };
    int left_flex() const { return marginals_[1].rows(); };
    int right_flex() const { return marginals_[1].cols(); };
    
    double scaler() const { return scaler_; };
    double& scaler() { return scaler_; };
    
    const Eigen::Ref<const Eigen::MatrixXd> marginal(int i) const { return marginals_[i]; };
    Eigen::Ref<Eigen::MatrixXd> marginal(int i) { return marginals_[i]; };
    
    const Eigen::Ref<const Eigen::MatrixXd> viterbi(int i) const { return viterbis_[i]; };
    Eigen::Ref<Eigen::MatrixXd> viterbi(int i) { return viterbis_[i]; };
    
    std::string label(int i) const { return labels_[i]; };
    std::string& label(int i) { return labels_[i]; };
    std::vector<std::string> labels() const { return labels_; };
};


/// @brief Smoosh two SmooshableStacks!
std::pair<SmooshableStack, std::vector<Eigen::MatrixXi>> SmooshStack(const SmooshableStack& s_a,
                                                                     const SmooshableStack& s_b);


}
