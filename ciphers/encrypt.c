/*COMPILE LIBRARY
 *Linux/Mac
 *gcc -shared -o encrypt.so -fPIC encrypt.c DES.c AES.c
 *Windows MinGW
 *gcc -shared -o encrypt.dll -Wl,--out-implib,libencrypt.a encrypt.c DES.c AES.c
*/
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "DES.h"
#include "AES.h"


/*
int main(int argc, const char * argv[]) {
    uint64_t input = 0xAAAAAAAAAAAAAAAA;
    uint64_t key = 0x0000000000000000;
    uint64_t result = input;
    int rounds= 1;
    
    result = des(input, key, rounds);
    printf ("DES encryption output: %016llx\n", result);
    
    exit(0);
}*/