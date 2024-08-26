# Aproximate pseudo-random function (based on block cyphers) to obtain a metrict on criptographic strongness based on neural network learning rate (eg: required dimension, time and samples)

The idea is to define a metric to compare different block cypers as a whole, not focusing on the singular substitution or permutation layer.
We would construct neural networks and train them to learn:
- one round encryption
- two round encryption
- three round encryption
- four round encryption (if computationally feasible)

The intuitive idea is that the bigger it is the network required and the slower it is to lear than most the criptographic function si strong. This can be used to compare different block cypers.

## Properties expected
- COMPARISON PROPERTY, the regression curv (exponential) calculated on 3 output points of a block cyper must alloways remain above or under the regression curv calculated on 3 output points of a different cypher
- NETWORK INDEPENDENCY, given 3 point representing the output afte 3 trainings (one for each number of rounds) and other 3 output points measured on a second different cryptographic function we want to mantain one above the other in the graph even if we recompute the same 6 points with different (maby bigger) neural networks

## First findings
The metod work on medium-difficulty sostitution-permutation network with 32bit block
