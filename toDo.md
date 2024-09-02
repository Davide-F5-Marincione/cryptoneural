# To do

- (g7240) insert new cipher with 128 bit block 
- (g7240) insert a stream cipher
- (g7240) check AES implementation to be sure it doesn't do to many rounds
- (davide-f5-marincione) insert ASCON in pythorch code
- (davide-f5-marincione) simulate ASCON to obtain data at round 1, 2, 3

## Done

- (g7240) create 64 bit block cryptographic function (DES) [the method must keep working even with bigger blocks otherwise it is useless]
- (g7240) create 128 bit block cryptographic function (AES) [real cryptographic function have 128bit block. The measure on them should seem higher than 64bit old ciphers]
- (davide-f5-marincione) test 64 bit crypto
- (davide-f5-marincione) test 128 bit crypto
- (davide-f5-marincione) define a bigger network for trying to learn four round encryption [At the moment, with 32bit block and 4 rounds, it is not learning. This could be due to the underdimensioned neural network or because the statistical correlation between input and output are too week. Because increasing encryption rounds make the problem exponentially harder it is reasonable to scale exponentially even the network dimension and the number of inputs. For these reason an attempt may be taken in account] --> defined multi cpu code
- (g7240) add an other famous 64bit block cipher (ASCON128v12) [for comparison reason]
- (g7240) found biggest parameters (now standard) to create the biggest neural network possible on a single 14GB GPU

### systematic definitive data production

- (g7240) tested with the biggest network DES 1, 2, 3; AES 1; AES 1 with Skipjack sbox; base 1, 2, 3 with AES sbox; base 1, 2, 3 with Skipjack sbox 

## Findings

- changing stronger sbox (like AES) with weaker sbox (Skipjack) doesn't influence the measure
- anyway the measure is not sbox agnostic: if inserted an sbox with some (<10%) fixed points (f(x)=x) the measure tells it is easier to be solved
- the biggest network possible with one 14GB GPU have this parameters n_layers=24 dim=4096 batch_size=8192