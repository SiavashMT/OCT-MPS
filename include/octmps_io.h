#ifndef OCTMPS_INCLUDE_OCTMPS_IO_H_
#define OCTMPS_INCLUDE_OCTMPS_IO_H_

#include "octmps.h"
#include "simulation.h"
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

void WriteInParm(FILE *, Simulation *);

double *AllocVector(short, short);

void WriteReflectanceClassIAndClassII(Simulation *,
                                        SimState *,
                                        FILE *,
                                        FILE *);

void WriteReflectanceClassIAndClassIIFiltered(Simulation *,
                                                 SimState *,
                                                 FILE *,
                                                 FILE *);

void WriteResult(Simulation *,
                 SimState *,
                 float *);

void FreeSimulation(Simulation *, int);

#ifdef __cplusplus
}   
#endif

#endif
