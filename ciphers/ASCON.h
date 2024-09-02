#ifndef ASCON_GUARD_H
#define ASCON_GUARD_H

// ascon 128 v12 (64bit block and 128bit key)
void ascon128_encrypt(unsigned char c[],unsigned char m[], unsigned char k[], int rounds);

#endif