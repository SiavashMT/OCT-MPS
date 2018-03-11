#include "octmps_io.h"

#include <float.h>
#include <time.h>
#include <stdio.h>
#include <math.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "octmps.h"
#include "mesh.h"
#include "simulation.h"

/***********************************************************
**  Allocate an array with index from nl to nh inclusive.
**
**  Original matrix and vector from Numerical Recipes in C
**  don't initialize the elements to zero. This will
**  be accomplished by the following functions.
************************************************************/
double *AllocVector(short nl, short nh) {
    double *v;
    short i;

    v = (double *) malloc((unsigned) (nh - nl + 1) * sizeof(double));
    if (!v) {
        fprintf(stderr, "allocation failure in vector()");
        exit(EXIT_FAILURE);
    }

    v -= nl;
    for (i = nl; i <= nh; i++) v[i] = 0.0;    /* init. */
    return v;
}

void WriteReflectanceClassIAndClassII(Simulation *in_ptr,
                                      SimState *out_ptr,
                                      FILE *output_file_classI_scatterings,
                                      FILE *output_file_classII_scatterings) {
    unsigned int depth_index;
    for (depth_index = 0; depth_index < in_ptr->num_optical_depth_length_steps; depth_index++) {

        // Class I parameters
        double mean_diffusive_reflectance = out_ptr->reflectance_classI_sum[depth_index]
                                            / in_ptr->number_of_photons;

        double variance_diffusive_reflectance_classI =
            (out_ptr->reflectance_classI_sum_sq[depth_index] / in_ptr->number_of_photons
             - mean_diffusive_reflectance * mean_diffusive_reflectance)
            / (in_ptr->number_of_photons - 1);

        // Class II parameters
        double mean_diffusive_reflectance_classII = out_ptr->reflectance_classII_sum[depth_index]
                / in_ptr->number_of_photons;

        double variance_diffusive_reflectance_classII
            = (out_ptr->reflectance_classII_sum_sq[depth_index] / in_ptr->number_of_photons
               - mean_diffusive_reflectance_classII
               * mean_diffusive_reflectance_classII)
              / (in_ptr->number_of_photons - 1);

        double optical_depth_temp = depth_index * in_ptr->coherence_length_source
                                    / NUM_SUBSTEPS_RESOLUTION
                                    + in_ptr->optical_depth_shift;

        fprintf(output_file_classI_scatterings, "%G,%G,%G,%G,%llu\n", in_ptr->probe_x, optical_depth_temp,
                mean_diffusive_reflectance,
                sqrt(variance_diffusive_reflectance_classI),
                out_ptr->num_classI_photons_filtered_in_range[depth_index]);

        fprintf(output_file_classII_scatterings, "%G,%G,%G,%G,%llu\n", in_ptr->probe_x, optical_depth_temp,
                mean_diffusive_reflectance_classII,
                sqrt(variance_diffusive_reflectance_classII),
                out_ptr->num_classII_photons_filtered_in_range[depth_index]);
    }
}

/*********************************************************************************
**  Write the reflectance of classes I and II Filtering up to one sample per bin
*********************************************************************************/
void WriteReflectanceClassIAndClassIIFiltered(Simulation *in_ptr,
        SimState *out_ptr,
        FILE *output_file_classI_scatterings,
        FILE *output_file_classII_scatterings) {

    long depth_index;
    for (depth_index = 0; depth_index < in_ptr->num_optical_depth_length_steps; depth_index++) {
        if (out_ptr->reflectance_classI_max[depth_index] > in_ptr->max_relative_contribution_to_bin_per_post_processed_sample
                * out_ptr->reflectance_classI_sum[depth_index]) {
            out_ptr->reflectance_classI_sum[depth_index] -= out_ptr->reflectance_classI_max[depth_index];
            out_ptr->reflectance_classI_sum_sq[depth_index] -= SQ(out_ptr->reflectance_classI_max[depth_index]);
            out_ptr->mean_reflectance_classI_sum[depth_index] -= out_ptr->reflectance_classI_max[depth_index];
            out_ptr->mean_reflectance_classI_sum_sq[depth_index] -= SQ(out_ptr->reflectance_classI_max[depth_index]);
        }

        if (out_ptr->reflectance_classII_max[depth_index] > in_ptr->max_relative_contribution_to_bin_per_post_processed_sample
                * out_ptr->reflectance_classII_sum[depth_index]) {
            out_ptr->reflectance_classII_sum[depth_index] -= out_ptr->reflectance_classII_max[depth_index];
            out_ptr->reflectance_classII_sum_sq[depth_index] -= SQ(out_ptr->reflectance_classII_max[depth_index]);
            out_ptr->mean_reflectance_classII_sum[depth_index] -= out_ptr->reflectance_classII_max[depth_index];
            out_ptr->mean_reflectance_classII_sum_sq[depth_index] -= SQ(out_ptr->reflectance_classII_max[depth_index]);
        }

        // Class I parameters
        double mean_diffusive_reflectance_classI = out_ptr->reflectance_classI_sum[depth_index]
                / in_ptr->number_of_photons;

        double variance_diffusive_reflectance
            = (out_ptr->reflectance_classI_sum_sq[depth_index] / in_ptr->number_of_photons
               - mean_diffusive_reflectance_classI * mean_diffusive_reflectance_classI)
              / (in_ptr->number_of_photons - 1);

        // Class II parameters
        double mean_diffusive_reflectance_classII = out_ptr->reflectance_classII_sum[depth_index]
                / in_ptr->number_of_photons;

        double variance_diffusive_reflectance_classII
            = (out_ptr->reflectance_classII_sum_sq[depth_index] / in_ptr->number_of_photons
               - mean_diffusive_reflectance_classII
               * mean_diffusive_reflectance_classII)
              / (in_ptr->number_of_photons - 1);

        double optical_depth_temp = depth_index * in_ptr->coherence_length_source
                                    / NUM_SUBSTEPS_RESOLUTION
                                    + in_ptr->optical_depth_shift;

        fprintf(output_file_classI_scatterings, "%G,%G,%G,%G,%llu\n",
                in_ptr->probe_x, optical_depth_temp,
                mean_diffusive_reflectance_classI,
                sqrt(variance_diffusive_reflectance),
                out_ptr->num_classI_photons_filtered_in_range[depth_index]);

        fprintf(output_file_classII_scatterings, "%G,%G,%G,%G,%llu\n",
                in_ptr->probe_x, optical_depth_temp,
                mean_diffusive_reflectance_classII,
                sqrt(variance_diffusive_reflectance_classII),
                out_ptr->num_classII_photons_filtered_in_range[depth_index]);
    }
}

void WriteResult(Simulation *in_params,
                 SimState *out_params,
                 float *time_report) {

    FILE *output_file_classI_scatterings = fopen("ClassI_Scatt.out", "a");
    FILE *output_file_classII_scatterings = fopen("ClassII_Scatt.out", "a");
    WriteReflectanceClassIAndClassII(in_params, out_params, output_file_classI_scatterings, output_file_classII_scatterings);
    fclose(output_file_classI_scatterings);
    fclose(output_file_classII_scatterings);

    output_file_classI_scatterings = fopen("ClassI_ScattFilt.out", "a");
    output_file_classII_scatterings = fopen("ClassII_ScattFilt.out", "a");
    WriteReflectanceClassIAndClassIIFiltered(in_params, out_params, output_file_classI_scatterings, output_file_classII_scatterings);
    fclose(output_file_classI_scatterings);
    fclose(output_file_classII_scatterings);
}

void FreeSimulation(Simulation *sim, int n_simulations) {
    int i;
    for (i = 0; i < n_simulations; i++)free(sim[i].regions);
    free(sim);
}
