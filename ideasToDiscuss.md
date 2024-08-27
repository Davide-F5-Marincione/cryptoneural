# List of ideas

## (1) Use sequentially chosen plain text values

### Idea

At the moment we are in a known-plaintext attack like model. We get at random $`x_i`$ and produce $`y_i`$. The $`x_1, \dots, x_n`$ plain text are not supposed to be correlated.

To help the network learn patterns and then speed up learning, we might simulate a chosen-plaintext attack scenario and pick correlated $`x_1, \dots, x_n`$ like $`x_1=199, x_2=200, x_3=201, x_4=202, x_5=203, \dots`$ The x are not taken
at random in this case so y pattern may be easier to be seen if the cryptographic function is weak.

If this experiment doesn't speed up learning then it is more general to remain in a known-plaintext attack like model.

### Discussion

This might be a bad idea. Although this may speed up the learning of the last part of the block and the mapping of its permutation layer part, this will also prevent a fast learning of the other blocks. For this reason I think this approach by itself may make the network training slower. However, it is still interesting to try to test it.
I think it would be a good idea starting providing only sequential correlated input at the beginning, to learn the underlying s-box and the fraction of p-box the subset of the input is subject to. Anyway this approach is not cipher agnostic because what we are trying to do is to variate only a subset of the input to learn the underling transformation the s-box perform on it.
For example, having cipher structured with this layer:
8bit_Sbox 8bit_Sbox 8bit_Sbox 8bit_Sbox
            32bit_Pbox
we may opt to test a train-set like

- 0x00000000 0x01000000 0x02000000 0x03000000 ... 0xFF000000, 0x00111111 0x01111111 0x02111111 0x03111111 ... 0xFF111111
- 0x00000000 0x00010000 0x00020000 0x00030000 ... 0x00FF0000, 0x11001111 0x11011111 0x11021111 0x11031111 ... 0x11FF1111
- 0x00000000 0x00000100 0x00000200 0x00000300 ... 0x0000FF00, ...
- 0x00000000 0x00000001 0x00000002 0x00000003 ... 0x000000FF, 0x11111100 0x11111101 0x11111102 0x11111103 ... 0x111111FF
- random samples until it finish to learn

This approach is really pour cipher agnostic, anyway if it bring to a faster learning it means that some correlation properties are exploited. To make it a little less chiper aweare we could simpli try:

- 0x10000000 0x20000000 .. 0xF0000000, 0x11000000 0x21000000 .. 0xF1000000, 0x12000000 0x22000000 .. 0xF2000000, ...
- 0x00000001 0x00000002 .. 0x0000000F, 0x00000011 0x00000012 .. 0x0000001F, 0x00000021 0x00000022 .. 0x0000002F, ...
- random samples until it finish to learn

More easy to implement schema:

- pic a random X, try training with approximately 256 or 512 X's consecutive elements, repeat this point again with a different random X
- finish training with each X chosen uniformly at random

We ended up with the initial idea as formulated at the beginning. We should try it and see.

### Test results

Not already tested.
