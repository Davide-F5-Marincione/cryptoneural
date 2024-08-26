## Use sequentially chosen plain text values
At the moment we are in a known-plaintext attack like model. We get at random $`x_i`$ and produce $`y_i`$. The $`x_1, \dots, x_n`$ plain text are not supposed to be correlated.

To help the network learn patterns and then speed up learning, we might simulate a chosen-plaintext attack schenario and pick correlated $`x_1, \dots, x_n`$ like $`x_1=199, x_2=200, x_3=201, x_4=202, x_5=203, \dots`$ The x are not taken 
at random in this case so y pattern may be easier to be seen if the cryptographic function is weak.

If this experiment doesen't speed up learning then it is more general to remain in a known-plaintext attack like model.
