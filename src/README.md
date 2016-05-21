# linearham                         {#mainpage}

The fundamental abstraction in linearham is a *smooshable*.
A smooshable is a linear segment of sequence that has a probability associated with each pair of begin and end points.
For example, a smooshable coming from a read aligned to a given germline sequence would have probability for a given pair of begin and end points equal to the probability of emitting the corresponding segment of read from that germline gene.
That is, it would be the probability of entering the linear segment of the HMM at the specified place, emitting the observed bases, and exiting at the specified place.

The allowable range for the begin and end points are called the *left flex* and *right flex*, respectively.
The begin and end points are indexed from the beginning of the smooshable.
An end point of zero corresponds to `right_flex` from the beginning, like so:

![](http://i.imgur.com/7CKeqed.png)

To *smoosh* two smooshables is to juxtapose two of them and consider all the options of going from the left one to the right one.
Here "consider" means both calculating marginal probabilities (summing over transition points) and calculating Viterbi paths.