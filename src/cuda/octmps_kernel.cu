#include "octmps_kernel.cuh"

#include <cuda_runtime.h>
#include <helper_cuda.h>


#include "cuda_rng.cuh"

__constant__ SimParamGPU d_simparam;
__constant__ RegionGPU d_region_specs[MAX_REGIONS];

__device__ void AtomicAddULL(unsigned long long* address, unsigned int add) {
#if __CUDA_ARCH__ > 110
	// CUDA only support attomic add for unsigned long long int in cc > 110
	// For lower versions use Emulated Atomic Add
	atomicAdd(address, (unsigned long long) add);
#else
	if (atomicAdd((unsigned int*)address, add) + add < add)
		atomicAdd(((unsigned int*)address) + 1, 1U);
#endif
}

__device__ double AtomicAddD(double* address, double value) {
	unsigned long long* address_as_ull = (unsigned long long*) address;
	unsigned long long old = *address_as_ull, assumed;
	do {
		assumed = old;
		old = atomicCAS(address_as_ull, assumed,
				__double_as_longlong(value + __longlong_as_double(assumed)));
	} while (assumed != old);
	return __longlong_as_double(old);
}

/**********************************************************
 **  Initialize Device Constant Memory with read-only data
 **********************************************************/
__host__ int InitDeviceConstantMemory(Simulation* sim, unsigned int root_index) {
	// Make sure that the number of regions is within the limit
	unsigned int n_regions = sim->n_regions;
	if (n_regions > MAX_REGIONS)
		return 1;

	SimParamGPU h_simparam;

	h_simparam.num_regions = sim->n_regions;
	h_simparam.r_specular = sim->r_specular;
	h_simparam.root_index = root_index;
	h_simparam.type_bias = sim->type_bias;
	h_simparam.target_depth_min = sim->target_depth_min;
	h_simparam.target_depth_max = sim->target_depth_max;
	h_simparam.backward_bias_coefficient = sim->backward_bias_coefficient;
	h_simparam.max_collecting_angle_deg = sim->max_collecting_angle_deg;
	h_simparam.max_collecting_radius = sim->max_collecting_radius;
	h_simparam.probability_additional_bias = sim->probability_additional_bias;
	h_simparam.optical_depth_shift = sim->optical_depth_shift;
	h_simparam.coherence_length_source = sim->coherence_length_source;
	h_simparam.num_optical_depth_length_steps = sim->num_optical_depth_length_steps;

	checkCudaErrors(cudaMemcpyToSymbol(d_simparam, &h_simparam, sizeof(SimParamGPU)));

	RegionGPU h_region_specs[MAX_REGIONS];

	for (unsigned int i = 0; i < n_regions; ++i) {

		h_region_specs[i].n = sim->regions[i].n;

		double rmuas = sim->regions[i].mutr;
		h_region_specs[i].muas = FP_ONE / rmuas;
		h_region_specs[i].rmuas = rmuas;
		h_region_specs[i].mua_muas = sim->regions[i].mua * rmuas;

		h_region_specs[i].g = sim->regions[i].g;
	}

	// Copy region data to constant device memory
	checkCudaErrors(cudaMemcpyToSymbol(d_region_specs, &h_region_specs, n_regions * sizeof(RegionGPU)));

	return 0;
}

/**************************************************************
 **  Transfer data from Device to Host memory after simulation
 **************************************************************/
__host__ int CopyDeviceToHostMem(SimState* host_mem,
								 SimState* device_mem, 
								 Simulation* sim, 
								 unsigned long long* d_x, 
								 unsigned long long* d_xR, 
								 unsigned long long* d_xS,
								 unsigned long long* d_num_classI_photons_filtered_in_range,
								 unsigned long long* d_num_classII_photons_filtered_in_range,
								 double* d_reflectance_classI_sum, 
								 double* d_reflectance_classI_max,
								 double* d_reflectance_classI_sum_sq,
								 double* d_reflectance_classII_sum,
								 double* d_reflectance_classII_max, 
								 double* d_reflectance_classII_sum_sq,
								 short steps) {

	checkCudaErrors(cudaMemcpy(&host_mem->probe_x, &device_mem->probe_x, sizeof(double), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(&host_mem->probe_y, &device_mem->probe_y, sizeof(double), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(&host_mem->probe_z, &device_mem->probe_z, sizeof(double), cudaMemcpyDeviceToHost));

	checkCudaErrors(cudaMemcpy(&host_mem->num_filtered_photons,
					&device_mem->num_filtered_photons, sizeof(unsigned long long),
					cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(&host_mem->num_filtered_photons_classI,
					&device_mem->num_filtered_photons_classI, sizeof(unsigned long long),
					cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(&host_mem->num_filtered_photons_classII,
					&device_mem->num_filtered_photons_classII, sizeof(unsigned long long),
					cudaMemcpyDeviceToHost));

	checkCudaErrors(cudaMemcpy(host_mem->num_classI_photons_filtered_in_range,
					d_num_classI_photons_filtered_in_range, steps * sizeof(unsigned long long),
					cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(host_mem->num_classII_photons_filtered_in_range,
					d_num_classII_photons_filtered_in_range, steps * sizeof(unsigned long long),
					cudaMemcpyDeviceToHost));

	checkCudaErrors(cudaMemcpy(host_mem->reflectance_classI_sum, d_reflectance_classI_sum,
					steps * sizeof(double), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(host_mem->reflectance_classI_max, d_reflectance_classI_max,
					steps * sizeof(double), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(host_mem->reflectance_classI_sum_sq,
					d_reflectance_classI_sum_sq, steps * sizeof(double),
					cudaMemcpyDeviceToHost));

	checkCudaErrors(cudaMemcpy(host_mem->reflectance_classII_sum,
					d_reflectance_classII_sum, steps * sizeof(double),
					cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(host_mem->reflectance_classII_max,
					d_reflectance_classII_max, steps * sizeof(double),
					cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(host_mem->reflectance_classII_sum_sq,
					d_reflectance_classII_sum_sq, steps * sizeof(double),
					cudaMemcpyDeviceToHost));

	return 0;
}

/*********************
 **  Free Host Memory
 *********************/
__host__ void FreeHostSimState(SimState* hstate) {
	if (hstate->n_photons_left != NULL) {
		free(hstate->n_photons_left);  // hstate->n_photons_left = NULL;
	}
	// TODO Compelete this
}

/********************
 **  Free GPU Memory
 ********************/
__host__ void FreeCudaMemory(int n, ...) {
	va_list arg_ptr;
	va_start(arg_ptr, n);

	for (int i=0; i<n; i++){
		void* ptr = va_arg(arg_ptr, void *);
		checkCudaErrors(cudaFree(ptr));
		ptr = NULL;
	}

	va_end(arg_ptr);
}

__device__ void CopyPhoton(PhotonGPU* original_photon, PhotonGPU* copied_photon) {
	copied_photon->x = original_photon->x;
	copied_photon->y = original_photon->y;
	copied_photon->z = original_photon->z;
	copied_photon->ux = original_photon->ux;
	copied_photon->uy = original_photon->uy;
	copied_photon->uz = original_photon->uz;
	copied_photon->w = original_photon->w;

	copied_photon->min_cos = original_photon->min_cos;
	copied_photon->root_index = original_photon->root_index;
	copied_photon->next_tetrahedron = original_photon->next_tetrahedron;

	copied_photon->s = original_photon->s;
	copied_photon->sleft = original_photon->sleft;

	copied_photon->optical_path = original_photon->optical_path;
	copied_photon->max_depth = original_photon->max_depth;
	copied_photon->likelihood_ratio = original_photon->likelihood_ratio;

	copied_photon->first_back_reflection_flag = original_photon->first_back_reflection_flag;
	copied_photon->depth_first_bias = original_photon->depth_first_bias;
	copied_photon->num_backwards_specular_reflections = original_photon->num_backwards_specular_reflections;
}

__device__ void LaunchPhoton(PhotonGPU* photon, double probe_x, double probe_y, double probe_z) {

	photon->x = probe_x;
	photon->y = probe_y;
	photon->z = probe_z;

	photon->ux = photon->uy = ZERO_FP;
	photon->uz = FP_ONE;
	photon->w = FP_ONE - d_simparam.r_specular;

	photon->hit = false;
	photon->min_cos = ZERO_FP;
	photon->s = ZERO_FP;
	photon->sleft = ZERO_FP;

	photon->optical_path = ZERO_FP;
	photon->max_depth = ZERO_FP;
	photon->likelihood_ratio = FP_ONE;
	photon->likelihood_ratio_after_first_bias = FP_ONE;

	photon->first_back_reflection_flag = false;
	photon->depth_first_bias = -FP_ONE;
	photon->num_backwards_specular_reflections = 0;

	photon->next_tetrahedron = -1;

	photon->root_index = d_simparam.root_index;
}

__device__ void ComputeStepSize(PhotonGPU* photon, unsigned long long* rnd_x,
		unsigned int* rnd_a) {
	// Make a new step if no leftover.
	if (photon->sleft == ZERO_FP) {
		double rand = rand_MWC_oc(rnd_x, rnd_a);
		photon->s = -log(rand) * d_region_specs[photon->tetrahedron[photon->root_index].region].rmuas;
	} else {
		photon->s = photon->sleft * d_region_specs[photon->tetrahedron[photon->root_index].region].rmuas;
		photon->sleft = ZERO_FP;
	}
}

__device__ bool HitBoundary(PhotonGPU* photon) {
	/* step size to boundary */
	unsigned int tetrahedron_index = photon->root_index;

	double min_distance = 1E10;
	int index_of_tetrahedron_with_min_distance = -1;

	double cos_normals_and_photon_direction[4];
	double distance_from_face_in_photon_direction;
	double perpendicular_distance;

	for (int i = 0; i < 4; i++) {
		int face_idx = photon->tetrahedron[tetrahedron_index].faces[i];
		int sign = photon->tetrahedron[tetrahedron_index].signs[i];
		cos_normals_and_photon_direction[i] = (photon->faces[face_idx].nx * sign) * photon->ux
				+ (photon->faces[face_idx].ny * sign) * photon->uy
				+ (photon->faces[face_idx].nz * sign) * photon->uz;
	}

	for (int i = 0; i < 4; i++) {
		if (cos_normals_and_photon_direction[i] < ZERO_FP) {
			// Photon is going toward the face
			int face_idx = photon->tetrahedron[tetrahedron_index].faces[i];
			int sign = photon->tetrahedron[tetrahedron_index].signs[i];
			perpendicular_distance = ((photon->faces[face_idx].nx * sign) * photon->x
					+ (photon->faces[face_idx].ny * sign) * photon->y
					+ (photon->faces[face_idx].nz * sign) * photon->z
					+ photon->faces[face_idx].d * sign);

			distance_from_face_in_photon_direction = (double) FAST_DIV(-perpendicular_distance, cos_normals_and_photon_direction[i]);

			if (distance_from_face_in_photon_direction < min_distance) {
				min_distance = distance_from_face_in_photon_direction;
				index_of_tetrahedron_with_min_distance = i;
				photon->min_cos = cos_normals_and_photon_direction[i];
			}
		}
	}

	if (photon->s > min_distance) {

		double mut = d_region_specs[photon->tetrahedron[tetrahedron_index].region].muas;
		photon->sleft = (photon->s - min_distance) * mut;
		photon->s = min_distance;
		photon->next_tetrahedron = index_of_tetrahedron_with_min_distance;
		return true;

	} else {
		photon->next_tetrahedron = -1;
		return false;
	}
}

__device__ void Hop(PhotonGPU* photon) {
	photon->x += photon->s * photon->ux;
	photon->y += photon->s * photon->uy;
	photon->z += photon->s * photon->uz;

	photon->optical_path += photon->s;
	if (photon->max_depth < photon->z)
		photon->max_depth = photon->z;
}

__device__ void Drop(PhotonGPU* photon) {
	unsigned int region = photon->tetrahedron[photon->root_index].region;
	double dwa = photon->w * d_region_specs[region].mua_muas;
	photon->w -= dwa;
}

__device__ void FastReflectTransmit(PhotonGPU* photon, SimState* d_state,
									unsigned long long* rnd_xR, unsigned int* rnd_aR, 
									double probe_x, double probe_y, double probe_z) {

	unsigned int root_index = photon->root_index;
	unsigned int ONE = 1;

	// cosines of transmission alpha
	double ux_reflected = 0, uy_reflected = 0, uz_reflected = 0;
	double ux_refracted = 0, uy_refracted = 0, uz_refracted = 0;

	// cosine of the incident angle (0 to 90 deg)
	double rFresnel;
	double ca1 = -(photon->min_cos);

	double ni = d_region_specs[photon->tetrahedron[root_index].region].n;
	double nt;
	int next = photon->next_tetrahedron;
	int next_tetrahedron_index = photon->tetrahedron[root_index].adjacent_tetrahedrons[next];
	if (next_tetrahedron_index == -1)
		nt = d_region_specs[0].n;  // Ambient medium's n
	else {
		int next_region = photon->tetrahedron[next_tetrahedron_index].region;
		nt = d_region_specs[next_region].n;
	}

	double ni_nt = (double) FAST_DIV(ni, nt);

	double sa1 = SQRT(FP_ONE - ca1 * ca1);
	if (ca1 > COSZERO)
		sa1 = ZERO_FP;
	double sa2 = FAST_MIN(ni_nt * sa1, FP_ONE);
	double ca2 = SQRT(FP_ONE - sa2 * sa2);

	double ca1ca2 = ca1 * ca2;
	double sa1sa2 = sa1 * sa2;
	double sa1ca2 = sa1 * ca2;
	double ca1sa2 = ca1 * sa2;

	// normal incidence: [(1-ni_nt)/(1+ni_nt)]^2
	// We ensure that ca1ca2 = 1, sa1sa2 = 0, sa1ca2 = 1, ca1sa2 = ni_nt
	if (ca1 > COSZERO) {
		sa1ca2 = FP_ONE;
		ca1sa2 = ni_nt;
	}

	double cam = ca1ca2 + sa1sa2; /* c- = cc + ss. */
	double sap = sa1ca2 + ca1sa2; /* s+ = sc + cs. */
	double sam = sa1ca2 - ca1sa2; /* s- = sc - cs. */

	rFresnel = (double) FAST_DIV(sam, sap * cam);
	rFresnel *= rFresnel;
	rFresnel *= (ca1ca2 * ca1ca2 + sa1sa2 * sa1sa2);

	// In this case, we do not care if "uz1" is exactly 0.
	if (ca1 < COSNINETYDEG || sa2 == FP_ONE)
		rFresnel = FP_ONE;

	unsigned int nxtFaceIdx = photon->tetrahedron[root_index].faces[photon->next_tetrahedron];
	int sign = photon->tetrahedron[root_index].signs[photon->next_tetrahedron];

	ux_refracted = photon->ux;
	uy_refracted = photon->uy;
	uz_refracted = photon->uz;

	ux_reflected = 2 * ca1 * photon->faces[nxtFaceIdx].nx * sign + photon->ux;
	uy_reflected = 2 * ca1 * photon->faces[nxtFaceIdx].ny * sign + photon->uy;
	uz_reflected = 2 * ca1 * photon->faces[nxtFaceIdx].nz * sign + photon->uz;

	double rand = rand_MWC_co(rnd_xR, rnd_aR);

	if (next_tetrahedron_index == -1 && rFresnel < FP_ONE) {  // -1 = NULL, no adjacent tetra
		photon->ux = ux_refracted;
		photon->uy = uy_refracted;
		photon->uz = uz_refracted;

		if (photon->uz < 0) {
			bool spatial_filter_flag = false;
			if (SQRT(SQ(photon->x-probe_x) + SQ(photon->y-probe_y)) < d_simparam.max_collecting_radius
					&& acos(-photon->uz) < d_simparam.max_collecting_angle_deg * PI_const / 180) {
				if (d_simparam.type_bias == 0 || d_simparam.type_bias == 3)
					spatial_filter_flag = true;

				if (photon->first_back_reflection_flag && d_simparam.type_bias == 37
						|| (photon->depth_first_bias == photon->max_depth && d_simparam.type_bias != 37)
								|| photon->num_backwards_specular_reflections > 0)
					spatial_filter_flag = true;
			}

			if (spatial_filter_flag) {
				AtomicAddULL(&d_state->num_filtered_photons, 1);
				double filter_optical_path = photon->optical_path;
				double filter_optical_depth = (double) FAST_DIV(filter_optical_path, FP_TWO);

				unsigned int vector_position = (unsigned int) (FAST_DIV(
						(FAST_DIV(filter_optical_path, FP_TWO) - d_simparam.optical_depth_shift),
						FAST_DIV(d_simparam.coherence_length_source, NUM_SUBSTEPS_RESOLUTION)));
				int ii;
				for (ii = vector_position - FAST_DIV(NUM_SUBSTEPS_RESOLUTION, 2);
						ii < vector_position + FAST_DIV(NUM_SUBSTEPS_RESOLUTION, 2);
						ii++) {
					if (ii >= 0 && ii < d_simparam.num_optical_depth_length_steps) {
						bool filter_classI_flag =
								photon->max_depth > (filter_optical_depth - FAST_DIV(d_simparam.coherence_length_source, FP_TWO));

						double temp_photon_contribution = photon->w * (FP_ONE - rFresnel) * photon->likelihood_ratio;

						if (filter_classI_flag) {
							if (ii == vector_position)
								AtomicAddULL(&d_state->num_filtered_photons_classI, ONE);

							AtomicAddD((&d_state->reflectance_classI_sum[ii]), temp_photon_contribution);

							if (d_state->reflectance_classI_max[ii] < temp_photon_contribution)
								d_state->reflectance_classI_max[ii] = temp_photon_contribution;

							AtomicAddD((&d_state->reflectance_classI_sum_sq[ii]),SQ(temp_photon_contribution));
							AtomicAddULL((&d_state->num_classI_photons_filtered_in_range[ii]), ONE);

						} else {
							if (ii == vector_position)
								AtomicAddULL(&d_state->num_filtered_photons_classII, ONE);

							if (d_state->reflectance_classII_max[ii] < temp_photon_contribution)
								d_state->reflectance_classII_max[ii] = temp_photon_contribution;

							AtomicAddD((&d_state->reflectance_classII_sum[ii]), temp_photon_contribution);
							AtomicAddD((&d_state->reflectance_classII_sum_sq[ii]), SQ(temp_photon_contribution));
							AtomicAddULL((&d_state->num_classII_photons_filtered_in_range[ii]), ONE);
						}
					}
				}
			}
		} else
			photon->num_backwards_specular_reflections++;

		// The rest of the photon will be reflected
		photon->w *= rFresnel;
		photon->ux = ux_reflected;
		photon->uy = uy_reflected;
		photon->uz = uz_reflected;

	} else if (rand > rFresnel) {

		photon->root_index = photon->tetrahedron[next_tetrahedron_index].index;

		photon->ux = ux_refracted;
		photon->uy = uy_refracted;
		photon->uz = uz_refracted;

		photon->next_tetrahedron = -1;

	} else {

		photon->next_tetrahedron = -1;

		if (photon->uz > ZERO_FP)
			photon->num_backwards_specular_reflections++;

		photon->ux = ux_reflected;
		photon->uy = uy_reflected;
		photon->uz = uz_reflected;

	}
}

__device__ double SpinTheta(double g, unsigned long long* rnd_x, unsigned int* rnd_a)
{
	double cost;

	if(g == ZERO_FP)
		cost = FP_TWO*rand_MWC_oc(rnd_x, rnd_a)-FP_ONE;
	else {
		double temp = (double) FAST_DIV((FP_ONE-SQ(g)),(FP_ONE - g + FP_TWO * g * rand_MWC_oc(rnd_x, rnd_a)));
		cost = (double) FAST_DIV((FP_ONE + SQ(g) - SQ(temp)),(FP_TWO * g));
		if(cost < -FP_ONE) cost = -FP_ONE;
		else if(cost > FP_ONE) cost = FP_ONE;
	}
	return(cost);
}

__device__ void Spin(double g, PhotonGPU* photon, unsigned long long* rnd_xS, unsigned int* rnd_aS) {
	double cost, sint;  // cosine and sine of the polar deflection angle theta
	double cosp, sinp;  // cosine and sine of the azimuthal angle psi
	double psi;
	double SIGN;
	double temp;
	double last_ux, last_uy, last_uz;

	/**************************************************************
	 **  Choose (sample) a new theta angle for photon propagation
	 **	according to the anisotropy.
	 **
	 **	If anisotropy g is 0, then
	 **		cos(theta) = 2*rand-1.
	 **	otherwise
	 **		sample according to the Henyey-Greenstein function.
	 **
	 **	Returns the cosine of the polar deflection angle theta.
	 ***************************************************************/

	cost = SpinTheta(g, rnd_xS, rnd_aS);
	sint = SQRT(FP_ONE - SQ(cost));

	psi = FP_TWO * PI_const * rand_MWC_co(rnd_xS, rnd_aS);;
	sincos(psi, &sinp, &cosp);

	double stcp = sint * cosp;
	double stsp = sint * sinp;

	last_ux = photon->ux;
	last_uy = photon->uy;
	last_uz = photon->uz;

	if (fabs(last_uz) > COSZERO) {  // Normal incident
		photon->ux = stcp;
		photon->uy = stsp;
		SIGN = ((last_uz) >= ZERO_FP ? FP_ONE : -FP_ONE);
		photon->uz = cost * SIGN;
	} else {  // Regular incident
		temp = RSQRT(FP_ONE - last_uz * last_uz);
		photon->ux = (stcp * last_ux * last_uz - stsp * last_uy) * temp
				+ last_ux * cost;
		photon->uy = (stcp * last_uy * last_uz + stsp * last_ux) * temp
				+ last_uy * cost;
		photon->uz = (double) FAST_DIV(-stcp, temp) + last_uz * cost;
	}
}

__device__ double SpinThetaForwardFstBias(double g, unsigned long long* rnd_x, unsigned int* rnd_a){
	double cost;

	if(g == ZERO_FP)
		cost = rand_MWC_oc(rnd_x, rnd_a);
	else {
		double randTmp = rand_MWC_oc(rnd_x, rnd_a);
		double temp = (double) FAST_DIV(randTmp, (FP_ONE-g)) + FAST_DIV((FP_ONE-randTmp), SQRT(SQ(g)+FP_ONE));
		cost = FAST_DIV(SQ(g) + FP_ONE - FAST_DIV(FP_ONE, SQ(temp) ),(FP_TWO*g));
		if(cost < -FP_ONE)
			cost = -FP_ONE;
		else if(cost > FP_ONE)
			cost = FP_ONE;
	}
	return(cost);
}

__device__ void SpinBias(double g, unsigned int region, 
						 PhotonGPU* photon, PhotonGPU* photon_cont, 
						 unsigned long long* rnd_x, unsigned int* rnd_a, 
						 double probe_x, double probe_y, double probe_z) {

	double g_squared = SQ(g);

	double cost, sint;  // cosine and sine of the polar deflection angle theta
	double cosp, sinp;  // cosine and sine of the azimuthal angle psi
	double costg, costg1, costg2;
	double psi;
	double temp;
	double ux = photon->ux;
	double uy = photon->uy;
	double uz = photon->uz;
	double ux_Orig = ux;
	double uy_Orig = uy;
	double uz_Orig = uz;
	double rand;

	double backward_bias_coefficient = d_simparam.backward_bias_coefficient;
	double bias_coefficient_temp = ZERO_FP;

	short reached_target_optical_depth_and_going_backwar_flag = 0;
	bool this_is_first_backward_bias_flag = false;

	if (photon->z > d_simparam.target_depth_min
			&& photon->z < d_simparam.target_depth_max
			&& photon->uz > ZERO_FP
			&& !photon->first_back_reflection_flag) {
		// This bias backwards will be applied only if the photon is going forward
		// The status of the photon prior to the bias will be saved

		CopyPhoton(photon, photon_cont);
		reached_target_optical_depth_and_going_backwar_flag = 1;
		photon->first_back_reflection_flag = true;  // Bias backwards only once
		photon->depth_first_bias = photon->z;
		bias_coefficient_temp = backward_bias_coefficient;
		this_is_first_backward_bias_flag = true;
	}

	/**********************************
	 ** Biased Direction towards probe
	 **********************************/
	double vx = probe_x-photon->x;
	double vy = probe_y-photon->y;
	double vz = probe_z-photon->z;
	double length_vector = SQRT(SQ(vx) + SQ(vy) + SQ(vz));
	vx = (double) FAST_DIV(vx, length_vector);
	vy = (double) FAST_DIV(vy, length_vector);
	vz = (double) FAST_DIV(vz, length_vector);
	/*********************************/

	if ((photon->first_back_reflection_flag
			|| photon->num_backwards_specular_reflections > 0)
			&& !reached_target_optical_depth_and_going_backwar_flag) {

		// It was biased at least once before and is moving backwards
		reached_target_optical_depth_and_going_backwar_flag = 2;

		double mut = d_region_specs[photon->tetrahedron[photon->root_index].region].muas;
		double next_step_size = (double) FAST_DIV(
				-log(rand_MWC_oc(rnd_x, rnd_a)), mut);
		double current_distance_to_probe = SQRT(
				SQ(photon->x-probe_x) + SQ(photon->y-probe_y) + SQ(photon->z-probe_z));

		if (next_step_size >= current_distance_to_probe
				&& acos(-vz) <= (d_simparam.max_collecting_angle_deg * PI_const / 180))
			reached_target_optical_depth_and_going_backwar_flag = 1;

		bias_coefficient_temp = backward_bias_coefficient;
	}

	short bias_function_randomly_selected = 0;
	if (reached_target_optical_depth_and_going_backwar_flag) {
		// Photon reached target optical region it may undergo additional biased
		// scattering or unbiased scattering

		// bias_function_randomly_selected=1 means use biased scattering and 2 means unbiased scattering
		rand_MWC_co(rnd_x, rnd_a) <= d_simparam.probability_additional_bias ? bias_function_randomly_selected = 1 : bias_function_randomly_selected = 2;

		if (reached_target_optical_depth_and_going_backwar_flag == 1
				|| bias_function_randomly_selected == 1) {
			/*************************************************************************
			 ** The photon is within the target depth and going forward
			 ** The additional biased scattering is randomly chosen
			 ** So the scattering is biased Henyey-Greenstein scattering
			 *************************************************************************/
			cost = SpinThetaForwardFstBias(bias_coefficient_temp, rnd_x, rnd_a);
			ux = vx;
			uy = vy;
			uz = vz;
		} else
			/**************************************************************************************
			 ** The photon is within the target depth but the scattering is randomly selected is
			 ** unbiased scattering
			 ** or the photon is already going backward or it is out of target depth
			 **************************************************************************************/
			cost = SpinTheta(g, rnd_x, rnd_a);

	} else {
		/**************************************************************************
		 **  The photon is not within the target depth or it is not going forward
		 **  so do unbiased scattering
		 **************************************************************************/
		cost = SpinTheta(g, rnd_x, rnd_a);
	}
	cost = FAST_MAX(cost, -FP_ONE);
	cost = FAST_MIN(cost, FP_ONE);

	sint = SQRT(FP_ONE - cost * cost);

	/* spin psi 0-2pi. */
	rand = rand_MWC_co(rnd_x, rnd_a);

	psi = FP_TWO * PI_const * rand;
	sincos(psi, &sinp, &cosp);

	double stcp = sint * cosp;
	double stsp = sint * sinp;

	if (fabs(uz) > COSZERO) {  // Normal incident
		photon->ux = stcp;
		photon->uy = stsp;
		photon->uz = copysign(cost, uz * cost);
	} else {  // Regular incident
		temp = RSQRT(FP_ONE - uz * uz);
		photon->ux = (stcp * ux * uz - stsp * uy) * temp + ux * cost;
		photon->uy = (stcp * uy * uz + stsp * ux) * temp + uy * cost;
		photon->uz = FAST_DIV(-stcp, temp) + uz * cost;
	}

	costg = ux_Orig * photon->ux
			+ uy_Orig * photon->uy
			+ uz_Orig * photon->uz;
	costg2 = costg;
	costg1 = vx * photon->ux
			+ vy * photon->uy
			+ vz * photon->uz;

	if (bias_coefficient_temp) {
		double one_plus_a_squared = 1 + SQ(bias_coefficient_temp);
		double sqrt_one_plus_a_squared = SQRT(one_plus_a_squared);
		double likelihood_ratio_increase_factor;
		if (reached_target_optical_depth_and_going_backwar_flag == 1)
			/****************************************************************************************
			 ** Likelihood for the first Bias scattering. Equation (8) of the paper:
			 ** Malektaji, Siavash, Ivan T. Lima, and Sherif S. Sherif. "Monte Carlo simulation of
			 ** optical coherence tomography for turbid media with arbitrary spatial distributions."
			 ** Journal of biomedical optics 19.4 (2014): 046001-046001.
			 ****************************************************************************************/
			likelihood_ratio_increase_factor = FAST_DIV(
					(1 - g_squared) * (sqrt_one_plus_a_squared - FP_ONE + bias_coefficient_temp) * SQRT( CUBE( one_plus_a_squared - FP_TWO * bias_coefficient_temp * cost)),
					FP_TWO*bias_coefficient_temp * (1-bias_coefficient_temp) * sqrt_one_plus_a_squared * SQRT( CUBE( FP_ONE+ g_squared - FP_TWO * g * costg))
			);

		else {
			double cost1, cost2;
			if (bias_function_randomly_selected == 1){
				cost1 = cost;
				cost2 = costg2;
			}
			else{
				cost1 = costg1;
				cost2 = cost;
			}
			/*******************************************************************************************************
			 **  The likelihood ratio of additional biased scatterings, whether the biased or the unbiased
			 **  probability density function is randomly selected, is calculated according to the equation (9)
			 **  of the paper:
			 **  Malektaji, Siavash, Ivan T. Lima, and Sherif S. Sherif. "Monte Carlo simulation of
			 **  optical coherence tomography for turbid media with arbitrary spatial distributions."
			 **  Journal of biomedical optics 19.4 (2014): 046001-046001.
			 *******************************************************************************************************/

			double pdf1 = (double) FAST_DIV(sqrt_one_plus_a_squared * bias_coefficient_temp * (FP_ONE - bias_coefficient_temp),
					(sqrt_one_plus_a_squared - FP_ONE + bias_coefficient_temp)*SQRT(CUBE(one_plus_a_squared - FP_TWO * bias_coefficient_temp * cost1)));

			double pdf2 = (double) FAST_DIV( FP_ONE - g_squared,
					(FP_TWO * CUBE(sqrt(FP_ONE + g_squared - FP_TWO * g * cost2))) );

			likelihood_ratio_increase_factor = (double) FAST_DIV( pdf2,
					(d_simparam.probability_additional_bias * pdf1 + (FP_ONE - d_simparam.probability_additional_bias)* pdf2) );
		}
		photon->likelihood_ratio *= likelihood_ratio_increase_factor;
		if (this_is_first_backward_bias_flag)
			// In case there was a sure backward bias and that was the very first one
			photon->likelihood_ratio_after_first_bias = photon->likelihood_ratio;

	}
}

__global__ void InitThreadState(GPUThreadStates* tstates,
								double probe_x, double probe_y, double probe_z) {
	
	PhotonGPU photon_temp;

	// This is the unique ID for each thread (or thread ID = tid)
	unsigned int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	unsigned int idy = (blockIdx.y * blockDim.y) + threadIdx.y;
	unsigned int tid = idx + idy * blockDim.x * gridDim.x;

	// Initialize the photon and copy into photon_<parameter x>
	LaunchPhoton(&photon_temp, probe_x, probe_y, probe_z);

	tstates->photon_x[tid] = photon_temp.x;
	tstates->photon_y[tid] = photon_temp.y;
	tstates->photon_z[tid] = photon_temp.z;
	tstates->photon_ux[tid] = photon_temp.ux;
	tstates->photon_uy[tid] = photon_temp.uy;
	tstates->photon_uz[tid] = photon_temp.uz;
	tstates->photon_w[tid] = photon_temp.w;
	tstates->hit[tid] = photon_temp.hit;
	tstates->min_cos[tid] = photon_temp.min_cos;

	tstates->photon_x_cont[tid] = photon_temp.x;
	tstates->photon_y_cont[tid] = photon_temp.y;
	tstates->photon_z_cont[tid] = photon_temp.z;
	tstates->photon_ux_cont[tid] = photon_temp.ux;
	tstates->photon_uy_cont[tid] = photon_temp.uy;
	tstates->photon_uz_cont[tid] = photon_temp.uz;
	tstates->photon_w_cont[tid] = photon_temp.w;
	tstates->min_cos_cont[tid] = photon_temp.min_cos;

	tstates->next_tetrahedron[tid] = photon_temp.next_tetrahedron;
	tstates->root_index[tid] = photon_temp.root_index;

	tstates->next_tetrahedron_cont[tid] = photon_temp.next_tetrahedron;
	tstates->root_index_cont[tid] = photon_temp.root_index;

	tstates->photon_s[tid] = photon_temp.s;
	tstates->photon_sleft[tid] = photon_temp.sleft;
	tstates->optical_path[tid] = photon_temp.optical_path;
	tstates->max_depth[tid] = photon_temp.max_depth;
	tstates->likelihood_ratio[tid] = photon_temp.likelihood_ratio;

	tstates->photon_s_cont[tid] = photon_temp.s;
	tstates->photon_sleft_cont[tid] = photon_temp.sleft;
	tstates->optical_path_cont[tid] = photon_temp.optical_path;
	tstates->max_depth_cont[tid] = photon_temp.max_depth;
	tstates->likelihood_ratio_cont[tid] = photon_temp.likelihood_ratio;


	tstates->likelihood_ratio_after_first_bias[tid] =
			photon_temp.likelihood_ratio_after_first_bias;

	tstates->first_back_reflection_flag[tid] = photon_temp.first_back_reflection_flag;
	tstates->depth_first_bias[tid] = photon_temp.depth_first_bias;
	tstates->num_backwards_specular_reflections[tid] =
			photon_temp.num_backwards_specular_reflections;

	tstates->first_back_reflection_flag_cont[tid] =
			photon_temp.first_back_reflection_flag;
	tstates->depth_first_bias_cont[tid] = photon_temp.depth_first_bias;
	tstates->num_backwards_specular_reflections_cont[tid] =
			photon_temp.num_backwards_specular_reflections;

	tstates->is_active[tid] = true;
}

__device__ void SaveThreadState(SimState* d_state, GPUThreadStates* tstates, 
								PhotonGPU* photon, PhotonGPU* photon_cont, 
								unsigned long long rnd_x, unsigned int rnd_a, 
								unsigned long long rnd_xR, unsigned int rnd_aR, 
								unsigned long long rnd_xS, unsigned int rnd_aS, 
								bool is_active) {

	unsigned int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	unsigned int idy = (blockIdx.y * blockDim.y) + threadIdx.y;
	unsigned int tid = idx + idy * blockDim.x * gridDim.x;


	d_state->x[tid] = rnd_x;
	d_state->a[tid] = rnd_a;
	d_state->xR[tid] = rnd_xR;
	d_state->aR[tid] = rnd_aR;
	d_state->xS[tid] = rnd_xS;
	d_state->aS[tid] = rnd_aS;

	tstates->photon_x[tid] = photon->x;
	tstates->photon_y[tid] = photon->y;
	tstates->photon_z[tid] = photon->z;
	tstates->photon_ux[tid] = photon->ux;
	tstates->photon_uy[tid] = photon->uy;
	tstates->photon_uz[tid] = photon->uz;
	tstates->photon_w[tid] = photon->w;
	tstates->hit[tid] = photon->hit;
	tstates->min_cos[tid] = photon->min_cos;

	tstates->photon_x_cont[tid] = photon_cont->x;
	tstates->photon_y_cont[tid] = photon_cont->y;
	tstates->photon_z_cont[tid] = photon_cont->z;
	tstates->photon_ux_cont[tid] = photon_cont->ux;
	tstates->photon_uy_cont[tid] = photon_cont->uy;
	tstates->photon_uz_cont[tid] = photon_cont->uz;
	tstates->photon_w_cont[tid] = photon_cont->w;
	tstates->min_cos_cont[tid] = photon_cont->min_cos;

	tstates->root_index[tid] = photon->root_index;
	tstates->next_tetrahedron[tid] = photon->next_tetrahedron;

	tstates->root_index_cont[tid] = photon_cont->root_index;
	tstates->next_tetrahedron_cont[tid] = photon_cont->next_tetrahedron;

	tstates->tetrahedron = photon->tetrahedron;
	tstates->faces = photon->faces;

	tstates->photon_s[tid] = photon->s;
	tstates->photon_sleft[tid] = photon->sleft;
	tstates->optical_path[tid] = photon->optical_path;
	tstates->max_depth[tid] = photon->max_depth;
	tstates->likelihood_ratio[tid] = photon->likelihood_ratio;

	tstates->photon_s_cont[tid] = photon_cont->s;
	tstates->photon_sleft_cont[tid] = photon_cont->sleft;
	tstates->optical_path_cont[tid] = photon_cont->optical_path;
	tstates->max_depth_cont[tid] = photon_cont->max_depth;
	tstates->likelihood_ratio_cont[tid] = photon_cont->likelihood_ratio;

	tstates->likelihood_ratio_after_first_bias[tid] =
			photon->likelihood_ratio_after_first_bias;

	tstates->first_back_reflection_flag[tid] = photon->first_back_reflection_flag;
	tstates->depth_first_bias[tid] = photon->depth_first_bias;
	tstates->num_backwards_specular_reflections[tid] =
			photon->num_backwards_specular_reflections;

	tstates->first_back_reflection_flag_cont[tid] =
			photon_cont->first_back_reflection_flag;
	tstates->depth_first_bias_cont[tid] = photon_cont->depth_first_bias;
	tstates->num_backwards_specular_reflections_cont[tid] =
			photon_cont->num_backwards_specular_reflections;

	tstates->is_active[tid] = is_active;
}

__device__ void CopyStatesToPhoton(GPUThreadStates* tstates, PhotonGPU* photon, PhotonGPU* photon_cont,
		                           bool* is_active, unsigned int tid) {

	/************
	 *  Photon
	 ***********/

	photon->x = tstates->photon_x[tid];
	photon->y = tstates->photon_y[tid];
	photon->z = tstates->photon_z[tid];

	photon->ux = tstates->photon_ux[tid];
	photon->uy = tstates->photon_uy[tid];
	photon->uz = tstates->photon_uz[tid];

	photon->w = tstates->photon_w[tid];
	photon->hit = tstates->hit[tid];

	photon->min_cos = tstates->min_cos[tid];

	photon->root_index = tstates->root_index[tid];
	photon->next_tetrahedron = tstates->next_tetrahedron[tid];

	photon->tetrahedron = tstates->tetrahedron;
	photon->faces = tstates->faces;

	photon->s = tstates->photon_s[tid];
	photon->sleft = tstates->photon_sleft[tid];
	photon->optical_path = tstates->optical_path[tid];
	photon->max_depth = tstates->max_depth[tid];
	photon->likelihood_ratio = tstates->likelihood_ratio[tid];

	photon->likelihood_ratio_after_first_bias =
			tstates->likelihood_ratio_after_first_bias[tid];

	photon->first_back_reflection_flag = tstates->first_back_reflection_flag[tid];
	photon->depth_first_bias = tstates->depth_first_bias[tid];

	photon->num_backwards_specular_reflections =
			tstates->num_backwards_specular_reflections[tid];

	/***************
	 *  Photon_Cont
	 ***************/

	photon_cont->x = tstates->photon_x_cont[tid];
	photon_cont->y = tstates->photon_y_cont[tid];
	photon_cont->z = tstates->photon_z_cont[tid];

	photon_cont->ux = tstates->photon_ux_cont[tid];
	photon_cont->uy = tstates->photon_uy_cont[tid];
	photon_cont->uz = tstates->photon_uz_cont[tid];

	photon_cont->w = tstates->photon_w_cont[tid];
	photon_cont->hit = tstates->hit[tid];  // Create tstates hit_cont

	photon_cont->min_cos = tstates->min_cos_cont[tid];

	photon_cont->root_index = tstates->root_index_cont[tid];
	photon_cont->next_tetrahedron = tstates->next_tetrahedron_cont[tid];

	photon_cont->tetrahedron = tstates->tetrahedron;
	photon_cont->faces = tstates->faces;

	photon_cont->s = tstates->photon_s_cont[tid];
	photon_cont->sleft = tstates->photon_sleft_cont[tid];
	photon_cont->optical_path = tstates->optical_path_cont[tid];
	photon_cont->max_depth = tstates->max_depth_cont[tid];
	photon_cont->likelihood_ratio = tstates->likelihood_ratio_cont[tid];

	photon_cont->first_back_reflection_flag =
			tstates->first_back_reflection_flag_cont[tid];
	photon_cont->depth_first_bias = tstates->depth_first_bias_cont[tid];

	photon_cont->num_backwards_specular_reflections =
			tstates->num_backwards_specular_reflections_cont[tid];

	*is_active = tstates->is_active[tid];
}

__device__ void RestoreThreadState(SimState* d_state, GPUThreadStates* tstates,
		PhotonGPU* photon, PhotonGPU* photon_cont, unsigned long long* rnd_x,
		unsigned int* rnd_a, unsigned long long* rnd_xR, unsigned int* rnd_aR, unsigned long long* rnd_xS,
		unsigned int* rnd_aS, bool* is_active, double* probe_x, double* probe_y, double* probe_z) {

	unsigned int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	unsigned int idy = (blockIdx.y * blockDim.y) + threadIdx.y;
	unsigned int tid = idx + idy * blockDim.x * gridDim.x;

	*rnd_x = d_state->x[tid];
	*rnd_a = d_state->a[tid];
	*rnd_xR = d_state->xR[tid];
	*rnd_aR = d_state->aR[tid];
	*rnd_xS = d_state->xS[tid];
	*rnd_aS = d_state->aS[tid];

	*probe_x = d_state->probe_x;
	*probe_y = d_state->probe_y;
	*probe_z = d_state->probe_z;

	CopyStatesToPhoton(tstates, photon, photon_cont, is_active, tid);
}

__global__ void OCTMPSKernel(SimState* d_state, GPUThreadStates* tstates) {
	// photon structure stored in registers
	PhotonGPU photon;
	PhotonGPU photon_cont;

	// random number seeds
	unsigned long long rnd_x, rnd_xR, rnd_xS;
	unsigned int rnd_a, rnd_aR, rnd_aS;

	// probe locations
	double probe_x, probe_y, probe_z;

	// Flag to indicate if this thread is active
	bool is_active;

	// Restore the thread state from global memory
	RestoreThreadState(d_state, tstates, &photon, &photon_cont, &rnd_x, &rnd_a,
			&rnd_xR, &rnd_aR, &rnd_xS, &rnd_aS, &is_active, &probe_x, &probe_y, &probe_z);

	for (int iIndex = 0; iIndex < NUM_STEPS; ++iIndex) {
		if (is_active) {

			ComputeStepSize(&photon, &rnd_x, &rnd_a);

			photon.hit = HitBoundary(&photon);

			Hop(&photon);

			if (photon.hit)
				FastReflectTransmit(&photon, d_state, &rnd_xR, &rnd_aR, probe_x, probe_y, probe_z);
			else {
				Drop(&photon);
				switch (d_simparam.type_bias) {
				case 0:
					Spin(d_region_specs[photon.tetrahedron[photon.root_index].region].g, &photon, &rnd_xS, &rnd_aS);
					break;
				case 37:
					SpinBias(d_region_specs[photon.tetrahedron[photon.root_index].region].g, photon.tetrahedron[photon.root_index].region, &photon, &photon_cont, &rnd_xS, &rnd_aS, probe_x, probe_y, probe_z);
					break;
				}
			}

			/***********************************************************
			 *  Roulette()
			 *  If the photon weight is small, the photon packet tries
			 *  to survive a roulette
			 ***********************************************************/
			if (photon.w < WEIGHT) {
				double rand = rand_MWC_co(&rnd_x, &rnd_a);
				if (photon.w != ZERO_FP && rand < CHANCE) 
					photon.w *= (double) FAST_DIV(FP_ONE, CHANCE);
				else if (photon.first_back_reflection_flag && d_simparam.type_bias != 3) {
					double likelihood_ratio_temp = photon.likelihood_ratio_after_first_bias;
					CopyPhoton(&photon_cont, &photon);
					Spin(d_region_specs[photon.tetrahedron[photon.root_index].region].g, &photon, &rnd_xS, &rnd_aS);
					photon.likelihood_ratio = (likelihood_ratio_temp < 1) ? 1 - likelihood_ratio_temp: 1;

				} else if (atomicSub(d_state->n_photons_left, 1) > gridDim.x * blockDim.x)
					LaunchPhoton(&photon, probe_x, probe_y, probe_z);
				else
					is_active = false;

			}
		}
	}
	__syncthreads();

	/**********************************************
	 ** Save the thread state to the global memory
	 **********************************************/
	SaveThreadState(d_state, tstates, &photon, &photon_cont, rnd_x, rnd_a,
			rnd_xR, rnd_aR, rnd_xS, rnd_aS, is_active);
}
