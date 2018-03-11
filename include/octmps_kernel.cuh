#ifndef OCTMPS_INCLUDE_KERNEL_CUH_
#define OCTMPS_INCLUDE_KERNEL_CUH_

#include <cuda_runtime.h>
#include <stdarg.h>

#include "octmps.h"
#include "octmps_kernel.h"
#include "simulation.h"

#define __MIN_CUDA_CC__ 350

#define NUM_THREADS_PER_BLOCK 192
#define EMULATED_ATOMIC

__device__ void AtomicAddULL(unsigned long long*, unsigned int);

__device__ double AtomicAddD(double*, double);

/**********************************************************
 **  Initialize Device Constant Memory with read-only data
 **********************************************************/
__host__ int InitDeviceConstantMemory(Simulation*, unsigned int);

/**************************************************************
 **  Transfer data from Device to Host memory after simulation
 **************************************************************/
__host__ int CopyDeviceToHostMem(SimState*, SimState*, Simulation*,
								 unsigned long long*, 
								 unsigned long long*,
								 unsigned long long*,  
								 unsigned long long*,
								 unsigned long long*,
								 double*, 
								 double*,
								 double*,
								 double*,
								 double*, 
								 double*,
								 short);

/*********************
 **  Free Host Memory
 *********************/
__host__ void FreeHostSimState(SimState*);

/********************
 **  Free GPU Memory
 ********************/
__host__ void FreeCudaMemory(int,...);

__device__ void CopyPhoton(PhotonGPU*, PhotonGPU*);

__device__ void LaunchPhoton(PhotonGPU*, double, double, double);

__device__ void ComputeStepSize(PhotonGPU*, unsigned long long*,
		unsigned int*);

__device__ bool HitBoundary(PhotonGPU*);

__device__ void Hop(PhotonGPU*);

__device__ void Drop(PhotonGPU*);

__device__ void FastReflectTransmit(PhotonGPU*, SimState*,
		unsigned long long*, unsigned int*, double, double, double);
	
__device__ double SpinTheta(double, unsigned long long*, unsigned int*);

__device__ void Spin(double, PhotonGPU*, unsigned long long*, unsigned int*);

__device__ double SpinThetaForwardFstBias(double, unsigned long long*, unsigned int*);

__device__ void SpinBias(double, unsigned int, PhotonGPU*, PhotonGPU*, unsigned long long*,
		unsigned int*, double, double, double);

__global__ void InitThreadState(GPUThreadStates*, double, double, double);

__device__ void SaveThreadState(SimState*, GPUThreadStates*,
		PhotonGPU*, PhotonGPU*, unsigned long long,
		unsigned int, unsigned long long, unsigned int, unsigned long long,
		unsigned int, bool);

__device__ void CopyStatesToPhoton(GPUThreadStates*,
		PhotonGPU*, PhotonGPU*,
		bool*, unsigned int);

__device__ void RestoreThreadState(SimState*, GPUThreadStates*,
		PhotonGPU*, PhotonGPU*, unsigned long long*,
		unsigned int*, unsigned long long*, unsigned int*, unsigned long long*,
		unsigned int*, bool*, double*, double*, double*);

__global__ void OCTMPSKernel(SimState*, GPUThreadStates*);

#endif 