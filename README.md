# Approximate pseudo-random function (based on block ciphers) to obtain a metric on cryptographic strength based on neural network learning rate (eg: required dimension, time and samples)

The idea is to define a metric to compare different block ciphers as a whole, not focusing on the singular substitution or permutation layer.
We would construct neural networks and train them to learn:

- one round encryption
- two round encryption
- three round encryption
- four round encryption (if computationally feasible)

The intuitive idea is that the bigger it is the network required and the slower it is to lear than most the cryptographic function si strong. This can be used to compare different block ciphers.

The measure that can be taken in account are the number of samples required for training, the network dimension and the required learning time to pass a threshold (eg: 0.9999). The most significant measure to be used seems to be  the training time because it depends both by network structure and number of samples required to learn the function cryptographic complexity.

## Properties expected

- SENSITIVITY, the measure must change (in a statistically significant way) as input cipher complexity changes
- COMPARISON PROPERTY, the regression curve (exponential) calculated on 3 output points of a block cipher must alloway remain above or under the regression curve calculated on 3 output points of a different cipher
- NETWORK INDEPENDENCY, given 3 point representing the output after 3 trainings (one for each number of rounds) and other 3 output points measured on a second different cryptographic function we want to maintain one above the other in the graph even if we recompute the same 6 points with different (maybe bigger) neural networks

## A possible concise cipher analysis output index: $`b`$

This might be an alternative to the graphical comparison of output plotted points.
As a concise comparison parameter proper of each cipher (but dependent by hardware speed) we might obtain the parameters $`b`$ obtained by the outputs interpolation whit the exponential function $`f(x)=a \cdot b^x`$. The function is expected to be an exponential function because increasing the round of encryption make the problem substantially more and more complex because of the explosion in the number of possible combinatorial states. To make  $`b`$ hardware independent we might consider to interpolate the points with y=time/hardwareSpeed instead of the simple y=time. Where hardwareSpeed could be measured in flops or experimentally by making all hardware solve a same problem and benchmark them on it.

P.S. A more reliable source for sure independent form hardware speed that could be used instead of training time  is the number of samples necessary to train the network.

P.P.S. To define a replicable measure it might be necessary to define fixed networks for rounds one, two, tree, four.

## First findings

The method work on medium-difficulty substitution-permutation network with 32bit block with three round encryption. The following plots represent two round encryption learning:

![plot](https://github.com/Davide-F5-Marincione/pseudoneural/blob/master/img/exampleImg1.jpeg)
![plot](https://github.com/Davide-F5-Marincione/pseudoneural/blob/master/img/exampleImg2.jpeg)
