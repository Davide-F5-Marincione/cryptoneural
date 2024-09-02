#ifndef DES_GUARD_H
# define DES_GUARD_H

#include <stdint.h>

uint64_t des(uint64_t input, uint64_t key, int rounds);

#endif