#ifndef OCTMPS_INCLUDE_OCTMPS_KERNEL_H_
#define OCTMPS_INCLUDE_OCTMPS_KERNEL_H_

#include "gpu_mesh.h"
#include "mesh.h"
#include "octmps.h"

#define FAST_DIV(x, y) __ddiv_rd(x,y) // __fdividef(x,y) // ((x)/(y))
#define ACCURATE_DIV(x, y) exp(log(x)-log(y))
#define SQRT(x) sqrt(x) // sqrtf(x)
#define RSQRT(x) rsqrt(x) // rsqrtf(x)
#define LOG(x) log(x) // logf(x)
#define SINCOS(x, sptr, cptr) sincos(x, sptr, cptr) // __sincosf(x, sptr, cptr)
#define SQ(x) ( (x)*(x) )
#define CUBE(x) ( (x)*(x)*(x) )
#define FAST_MIN(x, y) fmin(x, y)
#define FAST_MAX(x, y) fmax(x, y)

/*  Number of simulation steps performed by each thread in one kernel call */
#define NUM_STEPS 50000  //Use 5000 for faster response time

// The max number of regions supported (MAX_REGIONS including 1 ambient region)
#define MAX_REGIONS 100

typedef struct __align__(16){

    double r_specular;          // Specular Reflectance
    double target_depth_min;
    double target_depth_max;
    double backward_bias_coefficient;
    double max_collecting_angle_deg;
    double max_collecting_radius;
    double probability_additional_bias;
    double optical_depth_shift;
    double coherence_length_source;
    unsigned long long num_optical_depth_length_steps;
    unsigned int num_regions;        // number of regions
    unsigned int root_index;         // Root Tetrahedron Index
    short type_bias;
}
SimParamGPU;

typedef struct __align__(16) {

    double n;                  // refractive index of a region
    double muas;               // mua + mus
    double rmuas;              // 1/(mua+mus) = mutr = 1 / mua+mus
    double mua_muas;           // mua/(mua+mus)
    double g;                  // anisotropy
}
RegionGPU;

/*************************************************************************
** Thread-private states that live across batches of kernel invocations
** Each field is an array of length NUM_THREADS.
**
** We use a struct of arrays as opposed to an array of structs to enable
** global memory coalescing.
*************************************************************************/
typedef struct {
    int *next_tetrahedron;
    int *next_tetrahedron_cont;

    bool *is_active; // is this thread active?
    bool *hit;

    unsigned int *root_index;
    bool *first_back_reflection_flag;
    unsigned int *num_backwards_specular_reflections;

    unsigned int *root_index_cont;
    bool *first_back_reflection_flag_cont;
    unsigned int *num_backwards_specular_reflections_cont;

    // Cartesian coordinates of the photon [cm]
    double *photon_x;
    double *photon_y;
    double *photon_z;

    double *photon_x_cont;
    double *photon_y_cont;
    double *photon_z_cont;

    // directional cosines of the photon
    double *photon_ux;
    double *photon_uy;
    double *photon_uz;

    double *photon_ux_cont;
    double *photon_uy_cont;
    double *photon_uz_cont;

    double *photon_w;          // photon weight
    double *photon_s;          // photon step size
    double *photon_sleft;      // leftover step size [cm]

    double *photon_w_cont;     // photon weight
    double *photon_s_cont;     // photon step size
    double *photon_sleft_cont; // leftover step size [cm]

    double *min_cos;
    double *optical_path;
    double *max_depth;
    double *likelihood_ratio;
    double *depth_first_bias;

    double *min_cos_cont;
    double *optical_path_cont;
    double *max_depth_cont;
    double *likelihood_ratio_cont;
    double *depth_first_bias_cont;

    double *likelihood_ratio_after_first_bias;

    TetrahedronGPU *tetrahedron;
    TriangleFaces *faces;
} GPUThreadStates;

typedef struct {
    int next_tetrahedron;

    // flag to indicate if photon hits a boundary
    bool hit;

    bool first_back_reflection_flag;

    unsigned int root_index;
    unsigned int num_backwards_specular_reflections;

    // Cartesian coordinates of the photon [cm]
    double x;
    double y;
    double z;

    // directional cosines of the photon
    double ux;
    double uy;
    double uz;

    double w;            // photon weight

    double s;            // step size [cm]
    double sleft;        // leftover step size [cm]

    double min_cos;
    double depth_first_bias;
    double optical_path;
    double max_depth;
    double likelihood_ratio;
    double likelihood_ratio_after_first_bias;

    TetrahedronGPU *tetrahedron;
    TriangleFaces *faces;
} PhotonGPU;

#endif