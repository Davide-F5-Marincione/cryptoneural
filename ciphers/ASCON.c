/*Taken from https://csrc.nist.gov/projects/lightweight-cryptography/round-2-candidates zip source code ascon.zip/Implementations/crypto_aead/ascon128v12/opt64
Original code by: 
Christoph Dobraunig
Martin Schläffer*/

#include <stdio.h>
#include <stdlib.h>
#define CRYPTO_KEYBYTES 16
#define CRYPTO_NSECBYTES 0
#define CRYPTO_NPUBBYTES 16
#define CRYPTO_ABYTES 16
#define CRYPTO_NOOVERLAP 1
#include "ASCONendian.h"
#include "ASCONpermutations.h"

#define RATE (64 / 8)
#define PA_ROUNDS 12
#define PB_ROUNDS 6
#define IV                                                        \
  ((u64)(8 * (CRYPTO_KEYBYTES)) << 56 | (u64)(8 * (RATE)) << 48 | \
   (u64)(PA_ROUNDS) << 40 | (u64)(PB_ROUNDS) << 32)

#define Pr()            \
  if (rounds==1){       \
    P1();               \
  }                     \
  else if(rounds==2){   \
    P2();               \
  }                     \
  else if (rounds==3){  \
    P3();               \
  }                     \
  else if (rounds==4){  \
    P4();               \
  }                     \
  else{                 \
    printf("ASCON error. Selected to many rounds\n");  \
    exit(-1);           \
  }      

#define P2r()           \
  if (rounds==1){       \
    P2();               \
  }                     \
  else if(rounds==2){   \
    P4();               \
  }                     \
  else if (rounds==3){  \
    P6();               \
  }                     \
  else if (rounds==4){  \
    P8();               \
  }                     \
  else{                 \
    printf("ASCON error. Selected to many rounds\n");  \
    exit(-1);           \
  }           


int crypto128_aead_encrypt(unsigned char* c, unsigned long long* clen,           //ciphertext
                        const unsigned char* m, unsigned long long mlen,      //message
                        const unsigned char* ad, unsigned long long adlen,    //additional data
                        const unsigned char* nsec, const unsigned char* npub, //seconds or nonce as session freshness check
                        const unsigned char* k, int rounds) {                             //key
  const u64 K0 = U64BIG(*(u64*)k);
  const u64 K1 = U64BIG(*(u64*)(k + 8));
  const u64 N0 = U64BIG(*(u64*)npub);
  const u64 N1 = U64BIG(*(u64*)(npub + 8));
  state s;
  u64 i;
  (void)nsec;

  // set ciphertext size
  *clen = mlen + CRYPTO_ABYTES;

  // initialization
  s.x0 = IV;
  s.x1 = K0;
  s.x2 = K1;
  s.x3 = N0;
  s.x4 = N1;
  //P12();
  P2r()
  s.x3 ^= K0;
  s.x4 ^= K1;

  // process associated data
  if (adlen) {
    while (adlen >= RATE) {
      s.x0 ^= U64BIG(*(u64*)ad);
      //P6();
      Pr()
      adlen -= RATE;
      ad += RATE;
    }
    for (i = 0; i < adlen; ++i, ++ad) s.x0 ^= INS_BYTE64(*ad, i);
    s.x0 ^= INS_BYTE64(0x80, adlen);
    //P6();
    Pr()
  }
  s.x4 ^= 1;

  // process plaintext
  while (mlen >= RATE) {
    s.x0 ^= U64BIG(*(u64*)m);
    *(u64*)c = U64BIG(s.x0);
    //P6();
    Pr()
    mlen -= RATE;
    m += RATE;
    c += RATE;
  }
  for (i = 0; i < mlen; ++i, ++m, ++c) {
    s.x0 ^= INS_BYTE64(*m, i);
    *c = EXT_BYTE64(s.x0, i);
  }
  s.x0 ^= INS_BYTE64(0x80, mlen);

  // finalization
  s.x1 ^= K0;
  s.x2 ^= K1;
  //P12();
  P2r()
  s.x3 ^= K0;
  s.x4 ^= K1;

  // set tag
  *(u64*)c = U64BIG(s.x3);
  *(u64*)(c + 8) = U64BIG(s.x4);

  return 0;
}

void ascon128_encrypt(unsigned char c[],unsigned char m[], unsigned char k[], int rounds){
    if(rounds<=0 || rounds>3){
    printf("Error in ASCON rounds parameter (rounds=%d). It should be >0 and <=3\n",rounds);
    exit(-1);
  }
  const unsigned char* nsec= NULL;
  const unsigned char* ad= NULL; unsigned long long adlen= 0;    //additional data


  const unsigned char npub[16]= {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}; //seconds or nonce as session freshness check
  unsigned char authenticatedCipher[64+CRYPTO_ABYTES];
  unsigned long long clen[1]; //this is set by the function call
  unsigned long long mlen= 8; //8byte = 64bit message
  crypto128_aead_encrypt(authenticatedCipher, clen, m, mlen, ad, adlen, nsec, npub, k, rounds);
  for(int i=0; i<8; i++){
    c[i]= authenticatedCipher[i];
  }
  return;
}
/*
int main(){
  int i;
  unsigned char c[8];
  unsigned char m[8]= {1,2,3,4,5,6,7,8}; //64bit message
  unsigned char k[16]= {1,2,1,1,8,1,9,4,2,0,2,0,7,6,8,5};
  int rounds= 1;
  ascon128_encrypt(c, m, k, rounds);

  printf("c[8]= {");
  for(i=0; i<8; i++){
    printf("%u ",c[i]);
  }
  printf("}\n");
  return 0;
}*/
