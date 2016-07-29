
#include "smooshable_stack_chain.hpp"

/// @file smooshable_stack_chain.cpp
/// @brief Implementation of SmooshableStackChain class.

namespace linearham {


/// @brief Constructor for a SmooshableStackChain.
/// @param[in] originals
/// A vector of the input SmooshableStacks.
///
/// Does marginal and Viterbi calculation smooshing together a list of
/// SmooshableStacks.
SmooshableStackChain::SmooshableStackChain(std::vector<SmooshableStack> originals)
    : originals_(originals) {
  std::vector<IntMatrixVector> viterbi_idxs;

  // If there's only one SmooshableStack there is nothing to smoosh.
  if (originals.size() <= 1) {
    return;
  }
  
  // -------LAMBDA FUNCTION-------
  // Smoosh the supplied SmooshableStacks and add the results onto the back of the
  // corresponding vectors.
  auto SmooshAndAdd = [this, &viterbi_idxs](const SmooshableStack& s_a,
                                            const SmooshableStack& s_b) {
    SmooshableStack smooshed;
    IntMatrixVector viterbi_idx;
    std::tie(smooshed, viterbi_idx) = SmooshStack(s_a, s_b);
    
    smoosheds_.push_back(std::move(smooshed));
    viterbi_idxs.push_back(std::move(viterbi_idx));
  };
  
  // Say we are given SmooshableStacks a, b, c, and denote smoosh by *.
  // First make a list a*b, a*b*c.
  SmooshAndAdd(originals_[0], originals_[1]);
  for (unsigned int i = 2; i < originals_.size(); i++) {
    SmooshAndAdd(smoosheds_.back(), originals_[i]);
  };
  
  // In the example, this will be a*b*c.
  IntMatrixVector vidx_fully_smooshed = viterbi_idxs.back();
  
  // allocating storage for the viterbi paths
  viterbi_paths_.resize(vidx_fully_smooshed.size());
  
  // Unwind the viterbi paths.
  for (int k = 0; k < vidx_fully_smooshed.size(); k++) {
  
    for (int fs_i = 0; fs_i < vidx_fully_smooshed[k].rows(); fs_i++) {
      for (int fs_j = 0; fs_j < vidx_fully_smooshed[k].cols(); fs_j++) {
        std::vector<int> path;
        
        path.push_back(vidx_fully_smooshed[k](fs_i, fs_j));
        int trace_idx = k;
        std::vector<std::string>::iterator it;
        
        for (int l = (smoosheds_.size() - 2); l >= 0; l--) {
          
          it = std::find(smoosheds_[l].labels().begin(),
                         smoosheds_[l].labels().end(),
                         smoosheds_[l+1].label(trace_idx).substr(0, 2*l+3));
          trace_idx = it - smoosheds_[l].labels().begin();
          
          assert(path.front() < viterbi_idxs[l][trace_idx].cols());
          path.insert(path.begin(), viterbi_idxs[l][trace_idx](fs_i, path.front()));
          
        }
        
        viterbi_paths_[k].push_back(std::move(path));
      }
    }
    
  }
};

}
