#include "cuda_rng.cuh"

#include "octmps_kernel.h"

/****************************************************
 **  Generates a random number between 0 and 1 [0,1)
 ****************************************************/
__device__ double rand_MWC_co(unsigned long long* x, unsigned int* a) {
	//#define SEED0 985456376
	//#define SEED1 3858638025
	//#define SEED2 2658951225

	*x = (*x & 0xffffffffull) * (*a) + (*x >> 32);
	return FAST_DIV(__uint2float_rz((unsigned int )(*x)), (double )0x100000000);
	// The typecast will truncate the x so that it is 0<=x<(2^32-1),
	// __uint2double_rz ensures a round towards zero since 32-bit doubleing
	// point cannot represent all integers that large.
	// Dividing by 2^32 will hence yield [0,1)
}

/****************************************************
 **  Generates a random number between 0 and 1 (0,1]
 ****************************************************/
__device__ double rand_MWC_oc(unsigned long long* x, unsigned int* a) {
	return FP_ONE - rand_MWC_co(x, a);
}