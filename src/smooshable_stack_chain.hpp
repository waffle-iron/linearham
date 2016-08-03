
#include "smooshable_stack.hpp"

/// @file smooshable_stack_chain.hpp
/// @brief Headers for SmooshableStackChain class.

namespace linearham {



/// @brief An ordered list of SmooshableStacks that have been smooshed together, with
/// associated information.
///
/// The idea is that you put a collection of SmooshableStacks together in a chain
/// then smoosh them all together. It's nice to have a class for such a chain
/// so that you can unwind the result in the end.
class SmooshableStackChain {
 protected:
  std::vector<SmooshableStack> originals_;
  std::vector<SmooshableStack> smoosheds_;
  std::vector<IntVectorVector> viterbi_paths_;
  
 public:
  SmooshableStackChain(std::vector<SmooshableStack> originals);
  
  const std::vector<SmooshableStack>& originals() const { return originals_; };
  std::vector<SmooshableStack>& originals() { return originals_; };
  const std::vector<SmooshableStack>& smooshed() const { return smoosheds_; };
  std::vector<SmooshableStack>& smooshed() { return smoosheds_; };
  const std::vector<IntVectorVector>& viterbi_paths() const { return viterbi_paths_; };
  std::vector<IntVectorVector>& viterbi_paths() { return viterbi_paths_; };
};

}

