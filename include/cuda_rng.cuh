#ifndef OCTMPS_INCLUDE_CUDA_RNG_CUH_
#define OCTMPS_INCLUDE_CUDA_RNG_CUH_

#include "octmps.h"

/****************************************************
 **  Generates a random number between 0 and 1 [0,1)
 ****************************************************/
__device__ double rand_MWC_co(unsigned long long*, unsigned int*);

/****************************************************
 **  Generates a random number between 0 and 1 (0,1]
 ****************************************************/
__device__ double rand_MWC_oc(unsigned long long*, unsigned int*);

#endif
