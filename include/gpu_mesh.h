#ifndef OCTMPS_INCLUDE_GPU_MESH_H_
#define OCTMPS_INCLUDE_GPU_MESH_H_

typedef struct {
    unsigned int index, region;
    unsigned int faces[4], adjacent_tetrahedrons[4];
    short signs[4];
} TetrahedronGPU;

# endif
