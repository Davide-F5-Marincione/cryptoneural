/*COMPILE LIBRARY
 *Linux/Mac
 *gcc -shared -o encrypt.so -fPIC encrypt.c DES.c AES.c ASCON.c
 *Windows MinGW
 *gcc -shared -o miofile.dll -Wl,--out-implib,libmiofile.a miofile.c DES.c ASCON.c
*/
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "DES.h"
#include "AES.h"
#include "ASCON.h"


/*
int main(int argc, const char * argv[]) {
    //DES test
    uint64_t input = 0xAAAAAAAAAAAAAAAA;
    uint64_t key = 0x0000000000000000;
    uint64_t result = input;
    int rounds= 1;
    
    result = des(input, key, rounds);
    printf ("DES encryption output: %016llx\n", result);
    

    //ASCON test
    int i;
    unsigned char c[8];
    unsigned char m[8]= {1,2,3,4,5,6,7,8}; //64bit message
    unsigned char k[16]= {1,2,1,1,8,1,9,4,2,0,2,0,7,6,8,5};
    //int rounds= 1;
    ascon_encrypt(c, m, k, rounds);

    printf("c[8]= {");
    for(i=0; i<8; i++){
        printf("%u ",c[i]);
    }
    printf("}\n");

    exit(0);
}*/