#ifndef OCTMPS_INCLUDE_MESH_H_
#define OCTMPS_INCLUDE_MESH_H_

#include "octmps.h"
#include "simulation.h"

#ifdef __cplusplus
extern "C" {
#endif

// Data structure for specifying each region
typedef struct Region {
    double mutr;         // Reciprocal mu_total [cm]
    double mua;          // Absorption coefficient [1/cm]
    double mus;          // Scattering coefficient [1/cm]
    double g;            // Anisotropy factor [-]
    double n;            // Refractive index [-]
} Region;

typedef struct Vertex {
    unsigned int index;
    double x, y, z;
} Vertex;

typedef struct TriangleFaces {
    unsigned int index;
    double nx, ny, nz;
    double d;
    Vertex *vertices[3];
} TriangleFaces;

typedef struct Tetrahedron {
    unsigned int index;
    unsigned int region;
    Vertex *vertices[4];

    // The faces and adjacent_tetrahedrons should be in the same order. Meaning that the tetrahedron
    // and _adjacent_tetrahedrons[i] are common in face[i]
    TriangleFaces *faces[4];
    Tetrahedron *adjacent_tetrahedrons[4]; // By default this is NULL which means that there is no
    // Adjacent tetrahedron for that face (which means that
    // the tetrahedron is a surface tetrahedron)

} Tetrahedron;

inline double DotProduct(double, double, double, double, double, double);

Tetrahedron *FindRootofGraph(Tetrahedron *, Simulation **);

void SerializeGraph(Tetrahedron *, unsigned int, unsigned int, unsigned int,
                    Vertex *, TriangleFaces *, TetrahedronGPU *);

#ifdef __cplusplus
}
#endif

#endif
