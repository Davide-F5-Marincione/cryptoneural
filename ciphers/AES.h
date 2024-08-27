#ifndef AES_GUARD_H
# define AES_GUARD_H

char aes_encrypt(unsigned char *input, unsigned char *output, unsigned char *key, int size,  int effectiveRounds);

#endif
