#include "main.cuh"

#include <errno.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <float.h> // for FLT_MAX
#include <helper_cuda.h>
#include <helper_timer.h>
#include <pthread.h>
#include <time.h>

#include "octmps_kernel.h"
#include "simulation.h"

volatile int running_threads = 0;
volatile int wait_copy = 1;
pthread_mutex_t running_mutex = PTHREAD_MUTEX_INITIALIZER;

void FreeSimulation(Simulation* sim, int n_simulations) {
	int i;
	for(i=0;i<n_simulations;i++)
		free(sim[i].regions);
	free(sim);
}

void FreeMemory(int n, ...) {

	va_list arg_ptr;

	va_start(arg_ptr, n);

	for (int i=0; i<n; i++){
		void* ptr = va_arg(arg_ptr, void *);
		free(ptr);
		ptr = NULL;
	}

	va_end(arg_ptr);
}

/************************************
 **	Compute the specular reflection
 ************************************/
double RSpecular(double ni, double nt) {
	double r;
	double temp;
	temp = (ni - nt) / (ni + nt);
	r = temp * temp;
	return (r);
}

/***********************
** Initialize Host Mem
***********************/
void InitHostMem(SimState* &host_mem, short steps){

	unsigned int size;

	host_mem->num_filtered_photons = 0;
	host_mem->num_filtered_photons_classI = 0;
	host_mem->num_filtered_photons_classII = 0;

	size = steps * sizeof(unsigned long long);
	host_mem->num_classI_photons_filtered_in_range = (unsigned long long *) malloc(size);
	if (host_mem->num_classI_photons_filtered_in_range == NULL) {
		fprintf(stderr,"Error allocating host_mem->num_classI_photons_filtered_in_range");
		exit(EXIT_FAILURE);
	}
	for (int i = 0; i < steps; i++)
		host_mem->num_classI_photons_filtered_in_range[i] = 0;

	host_mem->num_classII_photons_filtered_in_range = (unsigned long long *) malloc(size);
	if (host_mem->num_classII_photons_filtered_in_range == NULL) {
		fprintf(stderr,"Error allocating host_mem->num_classII_photons_filtered_in_range");
		exit(EXIT_FAILURE);
	}
	for (int i = 0; i < steps; i++)
		host_mem->num_classII_photons_filtered_in_range[i] = 0;

	size = steps * sizeof(double);
	host_mem->reflectance_classI_sum = (double *) malloc(size);
	if (host_mem->reflectance_classI_sum == NULL) {
		fprintf(stderr,"Error allocating host_mem->reflectance_classI_sum");
		exit(EXIT_FAILURE);
	}
	for (int i = 0; i < steps; i++) 
		host_mem->reflectance_classI_sum[i] = 0.0;

	host_mem->reflectance_classI_max = (double *) malloc(size);
	if (host_mem->reflectance_classI_max == NULL) {
		fprintf(stderr,"Error allocating host_mem->reflectance_classI_max");
		exit(EXIT_FAILURE);
	}
	for (int i = 0; i < steps; i++) 
		host_mem->reflectance_classI_max[i] = 0.0;
	
	host_mem->reflectance_classI_sum_sq = (double *) malloc(size);
	if (host_mem->reflectance_classI_sum_sq == NULL) {
		fprintf(stderr,"Error allocating host_mem->reflectance_classI_sum_sq");
		exit(EXIT_FAILURE);
	}
	for (int i = 0; i < steps; i++) 
		host_mem->reflectance_classI_sum_sq[i] = 0.0;


	host_mem->reflectance_classII_sum = (double *) malloc(size);
	if (host_mem->reflectance_classII_sum == NULL) {
		fprintf(stderr,"Error allocating host_mem->reflectance_classII_sum");
		exit(EXIT_FAILURE);
	}
	for (int i = 0; i < steps; i++) 
		host_mem->reflectance_classII_sum[i] = 0.0; 
	

	host_mem->reflectance_classII_max = (double *) malloc(size);
	if (host_mem->reflectance_classII_max == NULL) {
		fprintf(stderr,"Error allocating host_mem->reflectance_classII_max");
		exit(EXIT_FAILURE);
	}
	for (int i = 0; i < steps; i++) 
		host_mem->reflectance_classII_max[i] = 0.0;
	
	host_mem->reflectance_classII_sum_sq = (double *) malloc(size);
	if (host_mem->reflectance_classII_sum_sq == NULL) {
		fprintf(stderr,"Error allocating host_mem->reflectance_classII_sum_sq");
		exit(EXIT_FAILURE);
	}
	for (int i = 0; i < steps; i++) 
		host_mem->reflectance_classII_sum_sq[i] = 0.0;
}

/**************************************************************
 **   Supports multiple GPUs
 **   Calls RunGPU with HostThreadState parameters
 ***************************************************************/
static CUT_THREADPROC RunGPUi(HostThreadState *hstate) {

	checkCudaErrors(cudaSetDevice(hstate->dev_id));
	cudaError_t cudastat;

	SimState *host_mem = &(hstate->host_sim_state);
	SimState *device_mem;

	GPUThreadStates *tstates;

	TetrahedronGPU *h_root = hstate->root;
	TriangleFaces *h_faces = hstate->faces;
	unsigned int *h_root_index = &(hstate->root_index);

	int region = hstate->root[*h_root_index].region;

	hstate->sim->r_specular = RSpecular(hstate->sim->regions[0].n, hstate->sim->regions[region].n);

	// total number of threads in the grid
	unsigned int n_threads = hstate->n_tblks * NUM_THREADS_PER_BLOCK;

	unsigned int n_tetrahedrons = hstate->n_tetrahedrons;
	unsigned int n_faces = hstate->n_faces;

	unsigned int size;

	short steps = hstate->sim->num_optical_depth_length_steps;

	InitHostMem(host_mem, steps);

	/***************************************
	**  Copy Host Memory to Device Memory 
	***************************************/
	unsigned int *d_n_photons_left;
    
    unsigned int *d_a, *d_aR, *d_aS;
    
    unsigned long long *d_x, *d_xR, *d_xS;

    unsigned long long *d_num_classI_photons_filtered_in_range, 
    				   *d_num_classII_photons_filtered_in_range;

    double *d_reflectance_classI_sum, *d_reflectance_classI_max,
           *d_reflectance_classI_sum_sq;

    double *d_reflectance_classII_sum, *d_reflectance_classII_max,
    	   *d_reflectance_classII_sum_sq;

    double *d_mean_reflectance_classI_sum, *d_mean_reflectance_classI_sum_sq,
    	   *d_mean_reflectance_classII_sum, *d_mean_reflectance_classII_sum_sq;

	size = n_threads * sizeof(SimState);
	checkCudaErrors(cudaMalloc((void **) &(device_mem), size));

	checkCudaErrors(cudaMemcpy(&(device_mem->probe_x), &hstate->sim->probe_x, sizeof(double), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(&(device_mem->probe_y), &hstate->sim->probe_y, sizeof(double), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(&(device_mem->probe_z), &hstate->sim->probe_z, sizeof(double), cudaMemcpyHostToDevice));

	size = sizeof(unsigned int);
	checkCudaErrors(cudaMalloc((void **) &d_n_photons_left, size));
	checkCudaErrors(cudaMemcpy(&(device_mem->n_photons_left), &d_n_photons_left, sizeof(unsigned int *), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_n_photons_left, host_mem->n_photons_left, size, cudaMemcpyHostToDevice));

	size = n_threads * sizeof(unsigned int);
	checkCudaErrors(cudaMalloc((void **) &d_a, size));
	checkCudaErrors(cudaMemcpy(&(device_mem->a), &d_a, sizeof(unsigned int *), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_a, host_mem->a, size, cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc((void **) &d_aR, size));
	checkCudaErrors(cudaMemcpy(&(device_mem->aR), &d_aR, sizeof(unsigned int *), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_aR, host_mem->aR, size, cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc((void **) &d_aS, size));
	checkCudaErrors(cudaMemcpy(&(device_mem->aS), &d_aS, sizeof(unsigned int *), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_aS, host_mem->aS, size, cudaMemcpyHostToDevice));

	size = n_threads * sizeof(unsigned long long);
	checkCudaErrors(cudaMalloc((void **) &d_x, size));
	checkCudaErrors(cudaMemcpy(&(device_mem->x), &d_x, sizeof(unsigned long long *), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_x, host_mem->x, size, cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc((void **) &d_xR, size));
	checkCudaErrors(cudaMemcpy(&(device_mem->xR), &d_xR, sizeof(unsigned long long *), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_xR, host_mem->xR, size, cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc((void **) &d_xS, size));
	checkCudaErrors(cudaMemcpy(&(device_mem->xS), &d_xS, sizeof(unsigned long long *),
			cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_xS, host_mem->xS, size, cudaMemcpyHostToDevice));

	size = sizeof(unsigned long long);
	checkCudaErrors(cudaMemset(&(device_mem->num_filtered_photons), 0, size));
	checkCudaErrors(cudaMemset(&(device_mem->num_filtered_photons_classI), 0, size));
	checkCudaErrors(cudaMemset(&(device_mem->num_filtered_photons_classII), 0, size));

	size *= steps;
	checkCudaErrors(cudaMalloc((void **) &d_num_classI_photons_filtered_in_range, size));
	checkCudaErrors(cudaMemcpy(&(device_mem->num_classI_photons_filtered_in_range),
			&d_num_classI_photons_filtered_in_range,
			sizeof(unsigned long long *), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemset(d_num_classI_photons_filtered_in_range, 0, size));

	checkCudaErrors(cudaMalloc((void **) &d_num_classII_photons_filtered_in_range, size));
	checkCudaErrors(cudaMemcpy(&(device_mem->num_classII_photons_filtered_in_range),
			&d_num_classII_photons_filtered_in_range,
			sizeof(unsigned long long *), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemset(d_num_classII_photons_filtered_in_range, 0, size));

	size = steps * sizeof(double);
	checkCudaErrors(cudaMalloc((void **) &d_reflectance_classI_sum, size));
	checkCudaErrors(cudaMemcpy(&(device_mem->reflectance_classI_sum),
			&d_reflectance_classI_sum,
			sizeof(double *), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemset(d_reflectance_classI_sum, 0, size));

	checkCudaErrors(cudaMalloc((void **) &d_reflectance_classI_max, size));
	checkCudaErrors(cudaMemcpy(&(device_mem->reflectance_classI_max),
			&d_reflectance_classI_max,
			sizeof(double *), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemset(d_reflectance_classI_max, 0, size));

	checkCudaErrors(cudaMalloc((void **) &d_reflectance_classI_sum_sq, size));
	checkCudaErrors(cudaMemcpy(&(device_mem->reflectance_classI_sum_sq),
			&d_reflectance_classI_sum_sq,
			sizeof(double *), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemset(d_reflectance_classI_sum_sq, 0, size));

	checkCudaErrors(cudaMalloc((void **) &d_reflectance_classII_sum, size));
	checkCudaErrors(cudaMemcpy(&(device_mem->reflectance_classII_sum),
			&d_reflectance_classII_sum,
			sizeof(double *), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemset(d_reflectance_classII_sum, 0, size));

	checkCudaErrors(cudaMalloc((void **) &d_reflectance_classII_max, size));
	checkCudaErrors(cudaMemcpy(&(device_mem->reflectance_classII_max),
			&d_reflectance_classII_max,
			sizeof(double *), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemset(d_reflectance_classII_max, 0, size));

	checkCudaErrors(cudaMalloc((void **) &d_reflectance_classII_sum_sq, size));
	checkCudaErrors(cudaMemcpy(&(device_mem->reflectance_classII_sum_sq),
			&d_reflectance_classII_sum_sq,
			sizeof(double *), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemset(d_reflectance_classII_sum_sq, 0, size));

	checkCudaErrors(cudaMalloc((void **) &d_mean_reflectance_classI_sum, size));
	checkCudaErrors(cudaMemcpy(&(device_mem->mean_reflectance_classI_sum),
			&d_mean_reflectance_classI_sum,
			sizeof(double *), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemset(d_mean_reflectance_classI_sum, 0, size));

	checkCudaErrors(cudaMalloc((void **) &d_mean_reflectance_classI_sum_sq, size));
	checkCudaErrors(cudaMemcpy(&(device_mem->mean_reflectance_classI_sum_sq),
			&d_mean_reflectance_classI_sum_sq,
			sizeof(double *), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemset(d_mean_reflectance_classI_sum_sq, 0, size));

	checkCudaErrors(cudaMalloc((void **) &d_mean_reflectance_classII_sum, size));
	checkCudaErrors(cudaMemcpy(&(device_mem->mean_reflectance_classII_sum),
			&d_mean_reflectance_classII_sum,
			sizeof(double *), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemset(d_mean_reflectance_classII_sum, 0, size));

	checkCudaErrors(cudaMalloc((void **) &d_mean_reflectance_classII_sum_sq, size));
	checkCudaErrors(cudaMemcpy(&(device_mem->mean_reflectance_classII_sum_sq),
			&d_mean_reflectance_classII_sum_sq,
			sizeof(double *), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemset(d_mean_reflectance_classII_sum_sq, 0, size));

	/***************************************
	**  Copy Host Memory to Thread State
	***************************************/

	int **d_next_tetrahedron,
		**d_next_tetrahedron_cont;

    bool **d_is_active,
    	 **d_hit;

    unsigned int **d_root_index;
    bool **d_first_back_reflection_flag;
    unsigned int **d_num_backwards_specular_reflections,
    			 **d_root_index_cont;

    bool **d_first_back_reflection_flag_cont;
    unsigned int **d_num_backwards_specular_reflections_cont;

    double *d_photon_x;
    double *d_photon_y;
    double *d_photon_z;

    double *d_photon_x_cont;
    double *d_photon_y_cont;
    double *d_photon_z_cont;

    double *d_photon_ux;
    double *d_photon_uy;
    double *d_photon_uz;

    double *d_photon_ux_cont;
    double *d_photon_uy_cont;
    double *d_photon_uz_cont;

    double *d_photon_w;
    double *d_photon_s;
    double *d_photon_sleft;

    double *d_photon_w_cont;
    double *d_photon_s_cont;
    double *d_photon_sleft_cont;

    double *d_min_cos;
    double *d_optical_path;
    double *d_max_depth;
    double *d_likelihood_ratio;
    double *d_depth_first_bias;

    double *d_min_cos_cont;
    double *d_optical_path_cont;
    double *d_max_depth_cont;
    double *d_likelihood_ratio_cont;
    double *d_depth_first_bias_cont;

    double *d_likelihood_ratio_after_first_bias;

    TetrahedronGPU *d_tetrahedron;
    TriangleFaces *d_triangleFaces;

	// Allocate tstates (photon structure) pointer on device
	size = n_threads * sizeof(GPUThreadStates);
	checkCudaErrors(cudaMalloc((void **) &(tstates), size));

	size = n_threads * sizeof(int);
	checkCudaErrors(cudaMalloc((void **) &d_next_tetrahedron, size));
	checkCudaErrors(cudaMemcpy(&(tstates->next_tetrahedron), &d_next_tetrahedron,
			sizeof(int *), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc((void **) &d_next_tetrahedron_cont, size));
	checkCudaErrors(cudaMemcpy(&(tstates->next_tetrahedron_cont), &d_next_tetrahedron_cont,
			sizeof(int *), cudaMemcpyHostToDevice));

	size = n_threads * sizeof(bool);
	checkCudaErrors(cudaMalloc((void **) &d_is_active, size));
	checkCudaErrors(cudaMemcpy(&(tstates->is_active), &d_is_active,
			sizeof(bool *), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc((void **) &d_hit, size));
	checkCudaErrors(cudaMemcpy(&(tstates->hit), &d_hit,
			sizeof(bool *), cudaMemcpyHostToDevice));

    size = n_threads * sizeof(unsigned int);
	checkCudaErrors(cudaMalloc((void **) &d_root_index, size));
	checkCudaErrors(cudaMemcpy(&(tstates->root_index), &d_root_index,
			sizeof(unsigned int *), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_root_index, h_root_index, size, cudaMemcpyHostToDevice));

    size = n_threads * sizeof(bool);
	checkCudaErrors(cudaMalloc((void **) &d_first_back_reflection_flag, size));
	checkCudaErrors(cudaMemcpy(&(tstates->first_back_reflection_flag), &d_first_back_reflection_flag,
			sizeof(bool *), cudaMemcpyHostToDevice));

    size = n_threads * sizeof(unsigned int);
	checkCudaErrors(cudaMalloc((void **) &d_num_backwards_specular_reflections, size));
	checkCudaErrors(cudaMemcpy(&(tstates->num_backwards_specular_reflections), &d_num_backwards_specular_reflections,
			sizeof(unsigned int *), cudaMemcpyHostToDevice));

    size = n_threads * sizeof(unsigned int);
	checkCudaErrors(cudaMalloc((void **) &d_root_index_cont, size));
	checkCudaErrors(cudaMemcpy(&(tstates->root_index_cont), &d_root_index_cont,
			sizeof(unsigned int *), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_root_index_cont, h_root_index, size,
			cudaMemcpyHostToDevice));

    size = n_threads * sizeof(bool);
	checkCudaErrors(cudaMalloc((void **) &d_first_back_reflection_flag_cont, size));
	checkCudaErrors(cudaMemcpy(&(tstates->first_back_reflection_flag_cont), &d_first_back_reflection_flag_cont,
			sizeof(bool *), cudaMemcpyHostToDevice));

    size = n_threads * sizeof(unsigned int);
	checkCudaErrors(cudaMalloc((void **) &d_num_backwards_specular_reflections_cont, size));
	checkCudaErrors(cudaMemcpy(&(tstates->num_backwards_specular_reflections_cont), &d_num_backwards_specular_reflections_cont,
			sizeof(unsigned int *), cudaMemcpyHostToDevice));


	size = n_threads * sizeof(double);
	checkCudaErrors(cudaMalloc((void **) &d_photon_x, size));
	checkCudaErrors(cudaMemcpy(&(tstates->photon_x), &d_photon_x,
			sizeof(double *), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc((void **) &d_photon_y, size));
	checkCudaErrors(cudaMemcpy(&(tstates->photon_y), &d_photon_y,
			sizeof(double *), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc((void **) &d_photon_z, size));
	checkCudaErrors(cudaMemcpy(&(tstates->photon_z), &d_photon_z,
			sizeof(double *), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc((void **) &d_photon_x_cont, size));
	checkCudaErrors(cudaMemcpy(&(tstates->photon_x_cont), &d_photon_x_cont,
			sizeof(double *), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc((void **) &d_photon_y_cont, size));
	checkCudaErrors(cudaMemcpy(&(tstates->photon_y_cont), &d_photon_y_cont,
			sizeof(double *), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc((void **) &d_photon_z_cont, size));
	checkCudaErrors(cudaMemcpy(&(tstates->photon_z_cont), &d_photon_z_cont,
			sizeof(double *), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc((void **) &d_photon_ux, size));
	checkCudaErrors(cudaMemcpy(&(tstates->photon_ux), &d_photon_ux,
			sizeof(double *), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc((void **) &d_photon_uy, size));
	checkCudaErrors(cudaMemcpy(&(tstates->photon_uy), &d_photon_uy,
			sizeof(double *), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc((void **) &d_photon_uz, size));
	checkCudaErrors(cudaMemcpy(&(tstates->photon_uz), &d_photon_uz,
			sizeof(double *), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc((void **) &d_photon_ux_cont, size));
	checkCudaErrors(cudaMemcpy(&(tstates->photon_ux_cont), &d_photon_ux_cont,
			sizeof(double *), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc((void **) &d_photon_uy_cont, size));
	checkCudaErrors(cudaMemcpy(&(tstates->photon_uy_cont), &d_photon_uy_cont,
			sizeof(double *), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc((void **) &d_photon_uz_cont, size));
	checkCudaErrors(cudaMemcpy(&(tstates->photon_uz_cont), &d_photon_uz_cont,
			sizeof(double *), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc((void **) &d_photon_w, size));
	checkCudaErrors(cudaMemcpy(&(tstates->photon_w), &d_photon_w,
			sizeof(double *), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc((void **) &d_photon_s, size));
	checkCudaErrors(cudaMemcpy(&(tstates->photon_s), &d_photon_s,
			sizeof(double *), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc((void **) &d_photon_sleft, size));
	checkCudaErrors(cudaMemcpy(&(tstates->photon_sleft), &d_photon_sleft,
			sizeof(double *), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc((void **) &d_photon_w_cont, size));
	checkCudaErrors(cudaMemcpy(&(tstates->photon_w_cont), &d_photon_w_cont,
			sizeof(double *), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc((void **) &d_photon_s_cont, size));
	checkCudaErrors(cudaMemcpy(&(tstates->photon_s_cont), &d_photon_s_cont,
			sizeof(double *), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc((void **) &d_photon_sleft_cont, size));
	checkCudaErrors(cudaMemcpy(&(tstates->photon_sleft_cont), &d_photon_sleft_cont,
			sizeof(double *), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc((void **) &d_min_cos, size));
	checkCudaErrors(cudaMemcpy(&(tstates->min_cos), &d_min_cos,
			sizeof(double *), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc((void **) &d_optical_path, size));
	checkCudaErrors(cudaMemcpy(&(tstates->optical_path), &d_optical_path,
			sizeof(double *), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc((void **) &d_max_depth, size));
	checkCudaErrors(cudaMemcpy(&(tstates->max_depth), &d_max_depth,
			sizeof(double *), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc((void **) &d_likelihood_ratio, size));
	checkCudaErrors(cudaMemcpy(&(tstates->likelihood_ratio), &d_likelihood_ratio,
			sizeof(double *), cudaMemcpyHostToDevice));

	size = n_threads * sizeof(double);
	checkCudaErrors(cudaMalloc((void **) &d_depth_first_bias, size));
	checkCudaErrors(cudaMemcpy(&(tstates->depth_first_bias), &d_depth_first_bias,
			sizeof(double *), cudaMemcpyHostToDevice));

	size = n_threads * sizeof(double);
	checkCudaErrors(cudaMalloc((void **) &d_min_cos_cont, size));
	checkCudaErrors(cudaMemcpy(&(tstates->min_cos_cont), &d_min_cos_cont,
			sizeof(double *), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc((void **) &d_optical_path_cont, size));
	checkCudaErrors(cudaMemcpy(&(tstates->optical_path_cont), &d_optical_path_cont,
			sizeof(double *), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc((void **) &d_max_depth_cont, size));
	checkCudaErrors(cudaMemcpy(&(tstates->max_depth_cont), &d_max_depth_cont,
			sizeof(double *), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc((void **) &d_likelihood_ratio_cont, size));
	checkCudaErrors(cudaMemcpy(&(tstates->likelihood_ratio_cont), &d_likelihood_ratio_cont,
			sizeof(double *), cudaMemcpyHostToDevice));

	size = n_threads * sizeof(double);
	checkCudaErrors(cudaMalloc((void **) &d_depth_first_bias_cont, size));
	checkCudaErrors(cudaMemcpy(&(tstates->depth_first_bias_cont), &d_depth_first_bias_cont,
			sizeof(double *), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc((void **) &d_likelihood_ratio_after_first_bias, size));
	checkCudaErrors(cudaMemcpy(&(tstates->likelihood_ratio_after_first_bias), &d_likelihood_ratio_after_first_bias,
			sizeof(double *), cudaMemcpyHostToDevice));

	size = n_tetrahedrons * sizeof(TetrahedronGPU);
	checkCudaErrors(cudaMalloc((void **) &d_tetrahedron, size));
	checkCudaErrors(cudaMemcpy(&(tstates->tetrahedron), &d_tetrahedron,
			sizeof(TetrahedronGPU *), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_tetrahedron, h_root, size, cudaMemcpyHostToDevice));
	
	 size = n_faces * sizeof(TriangleFaces);

	checkCudaErrors(cudaMalloc((void **) &d_triangleFaces, size));
	checkCudaErrors(cudaMemcpy(&(tstates->faces), &d_triangleFaces,
			sizeof(TriangleFaces *), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_triangleFaces, h_faces, size, cudaMemcpyHostToDevice));

	/******************************************************
	 **  Wait for all threads to finish copying above data
	 ******************************************************/
	checkCudaErrors(cudaThreadSynchronize());
	cudastat = cudaGetLastError(); // Check if there was an error
	if (cudastat) {
		fprintf(stderr,"[GPU %u] failure in InitSimStates (%i): %s\n",
				hstate->dev_id, cudastat, cudaGetErrorString(cudastat));
		FreeHostSimState(host_mem);
		FreeCudaMemory(6, d_a, d_aR, d_aS, d_x, d_xR, d_xS);
		FreeCudaMemory(18,
			d_root_index,
			d_num_backwards_specular_reflections,
			d_photon_x, d_photon_y, d_photon_z,
			d_photon_ux, d_photon_uy, d_photon_uz,
			d_photon_w, d_photon_s, d_photon_sleft,
			d_min_cos, d_optical_path, d_max_depth,
			d_likelihood_ratio, d_depth_first_bias, d_likelihood_ratio_after_first_bias,
			d_first_back_reflection_flag);
		FreeCudaMemory(18, d_next_tetrahedron_cont,
			d_root_index_cont,
			d_num_backwards_specular_reflections_cont,
			d_photon_x_cont, d_photon_y_cont, d_photon_z_cont,
			d_photon_ux_cont, d_photon_uy_cont, d_photon_uz_cont,
			d_photon_w_cont, d_photon_s_cont, d_photon_sleft_cont,
			d_min_cos_cont, d_optical_path_cont, d_max_depth_cont,
			d_likelihood_ratio_cont, d_depth_first_bias_cont,
			d_first_back_reflection_flag_cont);
		FreeCudaMemory(4, d_n_photons_left, d_hit, d_tetrahedron);
		exit(EXIT_FAILURE);
	}

	InitDeviceConstantMemory(hstate->sim, hstate->root_index);
	checkCudaErrors(cudaThreadSynchronize()); // Wait for all threads to finish
	cudastat = cudaGetLastError(); // Check if there was an error
	if (cudastat) {
		fprintf(stderr,"[GPU %u] failure in InitDCMem (%i): %s\n",
				hstate->dev_id, cudastat, cudaGetErrorString(cudastat));
		FreeHostSimState(host_mem);
        FreeCudaMemory(6, d_a, d_aR, d_aS, d_x, d_xR, d_xS);
		FreeCudaMemory(18,
			d_root_index,
			d_num_backwards_specular_reflections,
			d_photon_x, d_photon_y, d_photon_z,
			d_photon_ux, d_photon_uy, d_photon_uz,
			d_photon_w, d_photon_s, d_photon_sleft,
			d_min_cos, d_optical_path, d_max_depth,
			d_likelihood_ratio, d_depth_first_bias, d_likelihood_ratio_after_first_bias,
			d_first_back_reflection_flag);
		FreeCudaMemory(18, d_next_tetrahedron_cont,
			d_root_index_cont,
			d_num_backwards_specular_reflections_cont,
			d_photon_x_cont, d_photon_y_cont, d_photon_z_cont,
			d_photon_ux_cont, d_photon_uy_cont, d_photon_uz_cont,
			d_photon_w_cont, d_photon_s_cont, d_photon_sleft_cont,
			d_min_cos_cont, d_optical_path_cont, d_max_depth_cont,
			d_likelihood_ratio_cont, d_depth_first_bias_cont,
			d_first_back_reflection_flag_cont);
		FreeCudaMemory(4, d_n_photons_left, d_hit, d_tetrahedron);
		exit(EXIT_FAILURE);
	}

	dim3 dimBlock(NUM_THREADS_PER_BLOCK);
	dim3 dimGrid(hstate->n_tblks);

	// Initialize the remaining thread states
	InitThreadState << < dimGrid, dimBlock >> > (tstates, host_mem->probe_x, host_mem->probe_y, host_mem->probe_z);
	checkCudaErrors(cudaThreadSynchronize()); // Wait for all threads to finish
	cudastat = cudaGetLastError(); // Check if there was an error
	if (cudastat) {
		fprintf(stderr,"[GPU %u] failure in InitThreadState (%i): %s\n",
				hstate->dev_id, cudastat, cudaGetErrorString(cudastat));
		FreeHostSimState(host_mem);
		FreeCudaMemory(6, d_a, d_aR, d_aS, d_x, d_xR, d_xS);
		FreeCudaMemory(18,
			d_root_index,
			d_num_backwards_specular_reflections,
			d_photon_x, d_photon_y, d_photon_z,
			d_photon_ux, d_photon_uy, d_photon_uz,
			d_photon_w, d_photon_s, d_photon_sleft,
			d_min_cos, d_optical_path, d_max_depth,
			d_likelihood_ratio, d_depth_first_bias, d_likelihood_ratio_after_first_bias,
			d_first_back_reflection_flag);
		FreeCudaMemory(18, d_next_tetrahedron_cont,
			d_root_index_cont,
			d_num_backwards_specular_reflections_cont,
			d_photon_x_cont, d_photon_y_cont, d_photon_z_cont,
			d_photon_ux_cont, d_photon_uy_cont, d_photon_uz_cont,
			d_photon_w_cont, d_photon_s_cont, d_photon_sleft_cont,
			d_min_cos_cont, d_optical_path_cont, d_max_depth_cont,
			d_likelihood_ratio_cont, d_depth_first_bias_cont,
			d_first_back_reflection_flag_cont);
		FreeCudaMemory(4, d_n_photons_left, d_hit, d_tetrahedron);
		exit(EXIT_FAILURE);
	}

// Configure the L1 cache for Fermi
#ifdef USE_TRUE_CACHE
	cudaFuncSetCacheConfig(OCTMPSKernel, cudaFuncCachePreferL1);
#endif

	for (int i = 1; *host_mem->n_photons_left > 0; ++i) {

		// Run the kernel.
		OCTMPSKernel << < dimGrid, dimBlock >> > (device_mem, tstates);

		// Wait for all threads to finish.
		checkCudaErrors(cudaThreadSynchronize());

		// Check if there was an error
		cudastat = cudaGetLastError();
		if (cudastat) {
			fprintf(stderr,"[GPU %u] failure in OCTMPSKernel (%i): %s.\n",
					hstate->dev_id, cudastat, cudaGetErrorString(cudastat));
			FreeHostSimState(host_mem);

			exit(EXIT_FAILURE);
		}

		// Copy the number of photons left from device to host
		checkCudaErrors(cudaMemcpy(host_mem->n_photons_left, d_n_photons_left,
				sizeof(unsigned int), cudaMemcpyDeviceToHost));

		printf("[GPU %u] batch %5d, number of photons left %10u\n",
				hstate->dev_id, i, *(host_mem->n_photons_left));

		if (*host_mem->n_photons_left == 0) {
			pthread_mutex_lock(&running_mutex);
			running_threads--;
			pthread_mutex_unlock(&running_mutex);
		}
	}

	printf("[GPU %u] simulation done!\n", hstate->dev_id);

	wait_copy = CopyDeviceToHostMem(host_mem, device_mem, hstate->sim, 
									d_x, d_xR, d_xS,
									d_num_classI_photons_filtered_in_range,
									d_num_classII_photons_filtered_in_range,
									d_reflectance_classI_sum, 
									d_reflectance_classI_max,
									d_reflectance_classI_sum_sq,
									d_reflectance_classII_sum, 
									d_reflectance_classII_max,
									d_reflectance_classII_sum_sq,
									hstate->sim->num_optical_depth_length_steps);

	// We still need the host-side structure.
	FreeCudaMemory(6, d_a, d_aR, d_aS, d_x, d_xR, d_xS);
    FreeCudaMemory(18,
        d_root_index,
        d_num_backwards_specular_reflections,
        d_photon_x, d_photon_y, d_photon_z,
        d_photon_ux, d_photon_uy, d_photon_uz,
        d_photon_w, d_photon_s, d_photon_sleft,
        d_min_cos, d_optical_path, d_max_depth,
        d_likelihood_ratio, d_depth_first_bias, d_likelihood_ratio_after_first_bias,
        d_first_back_reflection_flag);
    FreeCudaMemory(18, d_next_tetrahedron_cont,
        d_root_index_cont,
        d_num_backwards_specular_reflections_cont,
        d_photon_x_cont, d_photon_y_cont, d_photon_z_cont,
        d_photon_ux_cont, d_photon_uy_cont, d_photon_uz_cont,
        d_photon_w_cont, d_photon_s_cont, d_photon_sleft_cont,
        d_min_cos_cont, d_optical_path_cont, d_max_depth_cont,
        d_likelihood_ratio_cont, d_depth_first_bias_cont,
        d_first_back_reflection_flag_cont);
    FreeCudaMemory(4, d_n_photons_left, d_hit, d_tetrahedron);
}

/*************************************************************************
 ** Perform OCTMPS simulation for one run out of N runs (in the input file)
 *************************************************************************/
static void DoOneSimulation(int sim_id,
		Simulation *simulation,
		TetrahedronGPU *d_root,
		TriangleFaces *d_faces,
		int n_tetrahedrons,
		int n_faces,
		int root_index,
		HostThreadState *hstates[],
		unsigned int num_GPUs,
		unsigned long long *x,
		unsigned int *a,
		unsigned long long *xR,
		unsigned int *aR,
		unsigned long long *xS,
		unsigned int *aS) {

	printf("\n------------------------------------------------------------\n");
	printf("        Simulation #%d\n", sim_id);
	printf("        - number_of_photons = %u\n", simulation->number_of_photons);
	printf("------------------------------------------------------------\n\n");

	// Distribute all photons among GPUs
	unsigned int n_photons_per_GPU = simulation->number_of_photons / num_GPUs;

	cudaDeviceProp props;
	int cc[num_GPUs];
	int num_pthreads = 0;
	for (int i = 0; i < num_GPUs; i++) {
		checkCudaErrors(cudaGetDeviceProperties(&props, hstates[i]->dev_id));
		cc[i] = ((props.major * 10 + props.minor) * 10);
		if (cc[i] != 110) num_pthreads++;
	}

	// For each GPU, init the host-side structure
	for (unsigned int i = 0; i < num_GPUs; ++i) {
		if (cc[i] != 110) {
			hstates[i]->sim = simulation;
			hstates[i]->n_tetrahedrons = n_tetrahedrons;
			hstates[i]->n_faces = n_faces;
			hstates[i]->root = d_root;
			hstates[i]->faces = d_faces;
			hstates[i]->root_index = root_index;
			SimState *hss = &(hstates[i]->host_sim_state);
			// number of photons responsible
			hss->n_photons_left = (unsigned int *) malloc(sizeof(unsigned int));
			// The last GPU may be responsible for more photons if the
			// distribution is uneven
			*(hss->n_photons_left) = (i == num_GPUs - 1) ?
					simulation->number_of_photons - (num_GPUs - 1) * n_photons_per_GPU :
					n_photons_per_GPU;
		}
	}
	// Start simulation kernel exec timer
	StopWatchInterface *execTimer = NULL;
	sdkCreateTimer(&execTimer);
	sdkStartTimer(&execTimer);

	// Launch simulation
	int failed = 0;
	// Launch a dedicated host thread for each GPU.
	pthread_t *hthreads;
	hthreads = (pthread_t *) malloc( sizeof(num_GPUs) * num_GPUs);
	for (unsigned int i = 0; i < num_GPUs; ++i) {
		if (cc[i] != 110) {
			pthread_mutex_lock(&running_mutex);
			running_threads++;
			pthread_mutex_unlock(&running_mutex);
			failed = pthread_create(&hthreads[i], NULL, (CUT_THREADROUTINE) RunGPUi, hstates[i]);
		}
	}
	// Wait for all host threads to finish
	timespec sleepValue = {0};
	sleepValue.tv_sec = 1;
	while (running_threads > 0) {
		nanosleep(&sleepValue, NULL);
	}
	for (unsigned int i = 0; i < num_GPUs; ++i) {
		if (cc[i] != 110) {
			pthread_join(hthreads[i], NULL);
		}
	}
	// Check any of the threads failed.
	for (unsigned int i = 0; i < num_GPUs && !failed; ++i) {
		if (cc[i] != 110) {
			if (hstates[i]->host_sim_state.n_photons_left == NULL) failed = 1;
		}
	}

	// End the timer
	printf("\nSimulation completed for all GPUs, stopping timer...\n");
	int timer = sdkStopTimer(&execTimer);
	printf("Timer stopped: %u - <1:true, 0:false>\n", timer);

	float elapsedTime = sdkGetTimerValue(&execTimer);
	printf("\n\n>>>>>>Simulation time: %f (ms)\n", elapsedTime);

	if (!failed) {
		// Sum the results to hstates[0]
		SimState *hss0 = &(hstates[0]->host_sim_state);
		short steps = hstates[0]->sim->num_optical_depth_length_steps;
		for (unsigned int i = 1; i < num_GPUs; i++) {
			SimState *hssi = &(hstates[i]->host_sim_state);
			hss0->num_filtered_photons += hssi->num_filtered_photons;
			hss0->num_filtered_photons_classI += hssi->num_filtered_photons_classI;
			hss0->num_filtered_photons_classII += hssi->num_filtered_photons_classII;

			for (int j = 0; j < steps; j++)
				hss0->num_classI_photons_filtered_in_range[j] +=
						hssi->num_classI_photons_filtered_in_range[j];
			for (int j = 0; j < steps; j++)
				hss0->num_classII_photons_filtered_in_range[j] +=
						hssi->num_classII_photons_filtered_in_range[j];
			for (int j = 0; j < steps; j++)
				hss0->reflectance_classI_sum[j] += hssi->reflectance_classI_sum[j];
			for (int j = 0; j < steps; j++)
				hss0->reflectance_classI_max[j] += hssi->reflectance_classI_max[j];
			for (int j = 0; j < steps; j++)
				hss0->reflectance_classI_sum_sq[j] += hssi->reflectance_classI_sum_sq[j];
			for (int j = 0; j < steps; j++)
				hss0->reflectance_classII_sum[j] += hssi->reflectance_classII_sum[j];
			for (int j = 0; j < steps; j++)
				hss0->reflectance_classII_max[j] += hssi->reflectance_classII_max[j];
			for (int j = 0; j < steps; j++)
				hss0->reflectance_classII_sum_sq[j] += hssi->reflectance_classII_sum_sq[j];
		}

		int size = steps * sizeof(double);

		hss0->mean_reflectance_classI_sum = (double *) malloc(size);
		if (hss0->mean_reflectance_classI_sum == NULL) {
			fprintf(stderr,"Error allocating hss0->mean_reflectance_classI_sum");
			exit(EXIT_FAILURE);
		}
		hss0->mean_reflectance_classI_sum_sq = (double *) malloc(size);
		if (hss0->mean_reflectance_classI_sum_sq == NULL) {
			fprintf(stderr,"Error allocating hss0->mean_reflectance_classI_sum_sq");
			exit(EXIT_FAILURE);
		}

		hss0->mean_reflectance_classII_sum = (double *) malloc(size);
		if (hss0->mean_reflectance_classII_sum == NULL) {
			fprintf(stderr,"Error allocating hss0->mean_reflectance_classII_sum");
			exit(EXIT_FAILURE);
		}
		hss0->mean_reflectance_classII_sum_sq = (double *) malloc(size);
		if (hss0->mean_reflectance_classII_sum_sq == NULL) {
			fprintf(stderr,"Error allocating hss0->mean_reflectance_classII_sum_sq");
			exit(EXIT_FAILURE);
		}
		for (unsigned long long jj = 0; jj < steps; jj++) {
			hss0->mean_reflectance_classI_sum[jj] = 0;
			hss0->mean_reflectance_classI_sum_sq[jj] = 0;
			hss0->mean_reflectance_classII_sum[jj] = 0;
			hss0->mean_reflectance_classII_sum_sq[jj] = 0;
		}
		WriteResult(simulation, hss0, &elapsedTime);
	}
	sdkDeleteTimer(&execTimer);

	// Free SimState structs
	for (unsigned int i = 0; i < num_GPUs; ++i) {
		if (cc[i] != 110) {
			FreeHostSimState(&(hstates[i]->host_sim_state));
		}
	}
}

int octmps_run(Tetrahedron* Graph,
		Simulation *simulations,
		unsigned int num_GPUs,
		int number_of_vertices,
		int number_of_faces,
		int number_of_tetrahedrons,
		int n_simulations) {

	// Determine the number of GPUs available
	int dev_count;
	checkCudaErrors(cudaGetDeviceCount(&dev_count));
	if (dev_count <= 0) {
		fprintf(stderr,"No GPU available. Quit.\n");
		return EXIT_FAILURE;
	}

	// Make sure we do not use more than what we have.
	if (num_GPUs > dev_count) {
		printf("The number of GPUs available is (%d).\n", dev_count);
		num_GPUs = (unsigned int) dev_count;
	}

	// Output the execution configuration
	printf("\n====================================\n");
	printf("EXECUTION MODE:\n");
	printf("  seed#1 Step + Roulette:  %llu\n", (unsigned long long)SEED0);
	printf("  seed#2 ReflectTransmit:  %llu\n", (unsigned long long)SEED1);
	printf("  seed#3 Spin:             %llu\n", (unsigned long long)SEED2);
	printf("  # of GPUs:               %u\n", num_GPUs);
	printf("====================================\n\n");

	printf("\n====================================\n");
	printf("  Number of tetrahedrons: %d\n", number_of_tetrahedrons);
	printf("  Number of vertices: %d\n", number_of_vertices);
	printf("  Number of faces: %d\n", number_of_faces);
	printf("====================================\n\n");

	Tetrahedron *root;
	Vertex *d_vertices = (Vertex *) malloc(sizeof(Vertex) * number_of_vertices);
	if (d_vertices == NULL){
		fprintf(stderr,"Could not allocate memory!");
		exit(EXIT_FAILURE);
	}
	TriangleFaces *d_faces = (TriangleFaces *) malloc(sizeof(TriangleFaces) * number_of_faces);
	if (d_faces == NULL){
		fprintf(stderr,"Could not allocate memory!");
		exit(EXIT_FAILURE);
	}
	TetrahedronGPU *d_root = (TetrahedronGPU *) malloc(sizeof(TetrahedronGPU) * number_of_tetrahedrons);
	if (d_root == NULL){
		fprintf(stderr,"Could not allocate memory!");
		exit(EXIT_FAILURE);
	}

	/**************************
	 ** B-Scan simulation loop
	 ***************************/
	// probe locations (currently only scanning in the x-axis direction)
	simulations->probe_y = 0;
	simulations->probe_z = 0;
	int n_Ascan = 1;
	int total_n_Ascans = int((simulations->probe->end_x - simulations->probe->start_x) /
	                                    simulations->probe->distance_Ascans);
	for (simulations->probe_x = simulations->probe->start_x;
			simulations->probe_x < simulations->probe->end_x;
			simulations->probe_x += simulations->probe->distance_Ascans) {

		printf("Finding the root of the graph...\n");

		root = FindRootofGraph(Graph, &simulations);
		unsigned int root_index = root->index;

		printf("Serializing the graph...\n");
		SerializeGraph(root, number_of_vertices, number_of_faces, number_of_tetrahedrons, d_vertices, d_faces, d_root);

        printf("\n------------------------------------------------------------\n");
        printf("Simulating A-Scan at x=%f, y=%f (A-Scan number %d out of %d A-Scans)\n", simulations->probe_x,
                simulations->probe_y, n_Ascan, total_n_Ascans);
        printf("------------------------------------------------------------\n\n");

		// Allocate one host thread state for each GPU
		HostThreadState **hstates;
		hstates = (HostThreadState **) malloc(sizeof(HostThreadState *) * num_GPUs);
		cudaDeviceProp props;
		int n_threads = 0;    // total number of threads for all GPUs
		int cc[num_GPUs];
		for (int i = 0; i < num_GPUs; ++i) {
			hstates[i] = (HostThreadState *) malloc(sizeof(HostThreadState));
			// Set the GPU ID
			hstates[i]->dev_id = i;
			// Get the GPU properties
			checkCudaErrors(cudaGetDeviceProperties(&props, hstates[i]->dev_id));
			// Validate the GPU compute capability
			cc[i] = (props.major * 10 + props.minor) * 10;
			if (cc[i] >= __MIN_CUDA_CC__) {
				printf("[GPU %u] \"%s\" with Compute Capability %d.%d (%d SMs)\n",
						i, props.name, props.major, props.minor,
						props.multiProcessorCount);
				// We launch one thread block for each SM on this GPU
				hstates[i]->n_tblks = props.multiProcessorCount;

				n_threads += hstates[i]->n_tblks * NUM_THREADS_PER_BLOCK;
			} else {
				fprintf(stderr,"\n[GPU %u] \"%s\" with Compute Capability %d.%d,"
						"\ndoes not meet the minimum requirement (1.3) for this program! "
						"\nExcluding [GPU %d].\n\n", i, props.name, props.major,
						props.minor, i);
			}
		}

		// Allocate and initialize RNG seeds (General, ReflectT, Spin)
		unsigned long long * x = (unsigned long long *) malloc(n_threads * sizeof(unsigned long long));
		unsigned int * a = (unsigned int *) malloc(n_threads * sizeof(unsigned int));

		unsigned long long * xR = (unsigned long long *) malloc(n_threads * sizeof(unsigned long long));
		unsigned int * aR = (unsigned int *) malloc(n_threads * sizeof(unsigned int));

		unsigned long long * xS = (unsigned long long *) malloc(n_threads * sizeof(unsigned long long));
		unsigned int * aS = (unsigned int *) malloc(n_threads * sizeof(unsigned int));

		if (InitRNG(x, a, xR, aR, xS, aS, n_threads, "safeprimes_base32.txt", (unsigned long long) SEED0))
			return EXIT_FAILURE;

		printf("\nUsing MWC random number generator ...\n");

		// Assign these seeds to each host thread state
		int ofst = 0;
		for (int i = 0; i < num_GPUs; ++i) {
			if (cc[i] != 110) {
				SimState *hss = &(hstates[i]->host_sim_state);
				hss->x = &x[ofst];
				hss->a = &a[ofst];
				hss->xR = &xR[ofst];
				hss->aR = &aR[ofst];
				hss->xS = &xS[ofst];
				hss->aS = &aS[ofst];
				ofst += hstates[i]->n_tblks * NUM_THREADS_PER_BLOCK;
			}
		}

		//perform all the simulations
		for (int i = 0; i < n_simulations; i++) {
			// Run a simulation
			DoOneSimulation(i, &simulations[i], d_root, d_faces,
					number_of_tetrahedrons, number_of_faces, root_index, hstates, num_GPUs,
					x, a, xR, aR, xS, aS);
		}

		// Free host thread states.
		for (int i = 0; i < num_GPUs; ++i) {
			free(hstates[i]);
		}

		// Free the random number seed arrays
		FreeMemory(6, x, a, xR, aR, xS, aS);
		n_Ascan++;
	}

	FreeSimulation(simulations, n_simulations);
	cudaDeviceReset();
	return EXIT_SUCCESS;
}
