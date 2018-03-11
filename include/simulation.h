#ifndef OCTMPS_INCLUDE_SIMULATION_H_
#define OCTMPS_INCLUDE_SIMULATION_H_

#include "gpu_mesh.h"
#include "octmps.h"

// Forward declaration for mesh.h due to circular includes
typedef struct Region Region;
typedef struct Vertex Vertex;
typedef struct TriangleFaces TriangleFaces;
typedef struct Tetrahedron Tetrahedron;

// Probe Position Data
typedef struct {
    double start_x;
    double end_x;
    double distance_Ascans;
} ProbeData;

// Simulation input parameters
typedef struct {

    // Current position of the probe
    double probe_x, probe_y, probe_z;
    // Probe information (A-scan distance, starting point etc)
    ProbeData *probe;

    unsigned int number_of_photons;

    // Not an input from user, it will be calculated in the code
    double r_specular;

    // Tetrahedrons variables
    unsigned short num_tetrahedrons;

    // Bias variables
    short type_bias;
    double backward_bias_coefficient;
    double coherence_length_source;

    double target_depth_min;
    double target_depth_max;

    double max_collecting_radius;
    double max_collecting_angle_deg;

    double probability_additional_bias;
    double max_relative_contribution_to_bin_per_post_processed_sample;

    // Not an input from user, it will be calculated in the code
    unsigned short num_optical_depth_length_steps;
    // Not an input from user, it will be calculated in the code
    double optical_depth_shift;

    unsigned int n_regions; // Number of regions including 1 region (with index 0) for ambient medium
    Region *regions;
} Simulation;

/***************************************************************************
** Per-GPU simulation states
** One instance of this struct exists in the host memory, while the other
** in the global memory.
***************************************************************************/
typedef struct {

    double probe_x, probe_y, probe_z;

    // points to a scalar that stores the number of photons that are not
    // completed (i.e. either on the fly or not yet started)
    unsigned int *n_photons_left;

    // per-thread seeds for random number generation
    // arrays of length NUM_THREADS
    // We put these arrays here as opposed to in GPUThreadStates because
    // they live across different simulation runs and must be copied back
    // to the host.

    unsigned int *a;  // General
    unsigned int *aR; // ReflectT
    unsigned int *aS; // Spin

    unsigned long long *x;  // General
    unsigned long long *xR; // ReflectT
    unsigned long long *xS; // Spin

    unsigned long long num_filtered_photons;            // Total number of filtered photons
    unsigned long long num_filtered_photons_classI;     // Total number of Class I filtered photons
    unsigned long long num_filtered_photons_classII;    // Total number of Class II filtered photons

    unsigned long long *num_classI_photons_filtered_in_range;   // Number of Class I filtered photons at each coherence gate
    unsigned long long *num_classII_photons_filtered_in_range;  // Number of Class II filtered photons at each coherence gate

    double *reflectance_classI_sum;
    double *reflectance_classI_max;
    double *reflectance_classI_sum_sq;

    double *reflectance_classII_sum;
    double *reflectance_classII_max;
    double *reflectance_classII_sum_sq;

    double *mean_reflectance_classI_sum;
    double *mean_reflectance_classI_sum_sq;
    double *mean_reflectance_classII_sum;
    double *mean_reflectance_classII_sum_sq;
} SimState;

// Everything a host thread needs to know in order to run simulation on
// one GPU (host-side only)
typedef struct {
    // GPU identifier
    unsigned int dev_id;

    // those states that will be updated
    SimState host_sim_state;

    // simulation input parameters
    Simulation *sim;

    // number of thread blocks launched
    unsigned int n_tblks;

    // root tetrahedron and faces
    TetrahedronGPU *root;
    TriangleFaces *faces;

    //number of tetrahedrons in actual use by RootTetrahedron
    unsigned int n_tetrahedrons;
    unsigned int n_faces;
    unsigned int root_index;
} HostThreadState;

#endif
