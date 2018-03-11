#ifndef OCTMPS_INCLUDE_MAIN_CUH_
#define OCTMPS_INCLUDE_MAIN_CUH_

#include <multithreading.h>

#include "octmps.h"
#include "octmps_io.h"
#include "octmps_kernel.cuh"

#include "rng.h"

void FreeSimulation(Simulation*, int);

void FreeMemory(int,...);

double Rspecular(double, double);

static CUT_THREADPROC RunGPUi(HostThreadState *);

static void DoOneSimulation(int,
                            Simulation *,
                            TetrahedronGPU *,
                            TriangleFaces *,
                            int ,
                            int ,
                            int ,
                            HostThreadState *[],
                            unsigned int ,
                            unsigned long long *,
                            unsigned int *,
                            unsigned long long *,
                            unsigned int *,
                            unsigned long long *,
                            unsigned int *);

int octmps_run(Tetrahedron*,
		       Simulation *,
		       unsigned int,
		       int,
		       int,
		       int,
		       int);

#endif