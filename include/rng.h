#ifndef OCTMPS_INCLUDE_RNG_H_
#define OCTMPS_INCLUDE_RNG_H_

#include "octmps.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SEED0 985456376
#define SEED1 3858638025
#define SEED2 2658951225

int InitRNG(unsigned long long*, unsigned int*, unsigned long long*, unsigned int*, unsigned long long*,
		unsigned int*, const unsigned int, const char*,
		unsigned long long);

#ifdef __cplusplus
}
#endif


#endif
