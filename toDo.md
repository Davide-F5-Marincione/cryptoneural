# To do

- (g7240) make cryptographic functions fast producing n (eg. 384) consecutive elements encryption at a time [at the moment those are the CPU bottleneck]
- (g7240) add an other famous 64bit block cipher [for comparison reason]
- (davide-f5-marincione) test 64 bit crypto
- (g7240) create 128 bit block cryptographic function [real cryptographic function have 128bit block. The measure on them should seem higher than 64bit old ciphers]
- (davide-f5-marincione) test 128 bit crypto
- (davide-f5-marincione) define a bigger network for trying to learn four round encryption [At the moment, with 32bit block and 4 rounds, it is not learning. This could be due to the underdimensioned neural network or because the statistical correlation between input and output are too week. Because increasing encryption rounds make the problem exponentially harder it is reasonable to scale exponentially even the network dimension and the number of inputs. For these reason an attempt may be taken in account]

## Done

- (g7240) create 64 bit block cryptographic function (DES) [the method must keep working even with bigger blocks otherwise it is useless]
