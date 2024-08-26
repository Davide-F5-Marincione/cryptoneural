# Aproximate pseudo-random function (based on block cyphers) to obtain a metrict on criptographic strongness based on neural network learning rate (eg: required dimension, time and samples)

The idea is to define a metric to compare different block cypers as a whole, not focusing on the singular substitution or permutation layer.
We would construct neural networks and train them to learn:
- one round encryption
- two round encryption
- three round encryption
- four round encryption (if computationally feasible)

The intuitive idea is that the bigger it is the network required and the slower it is to lear than most the criptographic function si strong. This can be used to compare different block cypers. 

The measure that can be taken in account are the number of samples required for training, the network dimension and the required learning time to pass a threshold (eg: 0.9999). The most significant measure to be used seems to be  the training time because it depends both by network structure and number of semples required to learn the function cryptographic complexity.


## Properties expected
- SENSITIVITY, the measure must change (in a statistically significant way) as input cypher complexity changes
- COMPARISON PROPERTY, the regression curv (exponential) calculated on 3 output points of a block cyper must alloways remain above or under the regression curv calculated on 3 output points of a different cypher
- NETWORK INDEPENDENCY, given 3 point representing the output afte 3 trainings (one for each number of rounds) and other 3 output points measured on a second different cryptographic function we want to mantain one above the other in the graph even if we recompute the same 6 points with different (maby bigger) neural networks


## A possible concise cypher analisys output index: $`b`$
This might be an alternative to the graphical comparison of output plotted points.
As a concise comparison parameter proper of each cypher (but dependent by hardware speed) we might obtain the parameters $`b`$ obtained by the outputs interpolation whith the exponential function $`f(x)=a \cdot b^x`$. The function is expected to be an exponential function because increasing the round of encryption make the poblem substantially more and more complex because of the explosion in the number of possible combinatorial states. To make  $`b`$ hardware independent we myght consider to interpolate the points with y=time/hardwareSpeed instead of the simple y=time. Where hardwareSpeed could be measured in flops or esperimentally by making all hardwars solve a same problem and beanchmark them on it. 

P.S. A more reliable souce for shure independent form hardware speed that could be used instead of training time  is the number of samples necessary to train the network.

P.P.S. To define a replicable measure it might be necessary to define fixed networks for rounds one, two, tree, four.


## First findings
The metod work on medium-difficulty sostitution-permutation network with 32bit block with three round encryption. The following plots represent two round encryption learning:

![plot](https://github.com/Davide-F5-Marincione/pseudoneural/blob/master/img/exampleImg1.jpeg) 
![plot](https://github.com/Davide-F5-Marincione/pseudoneural/blob/master/img/exampleImg2.jpeg)
