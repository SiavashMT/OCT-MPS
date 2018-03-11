#include "mesh.h"

#include <math.h>
#include <stdbool.h>
#include <stdio.h>

inline double DotProduct(double x1, double y1, double z1, double x2, double y2, double z2) {
	return x1 * x2 + y1 * y2 + z1 * z2;
}

/****************************************
 **  Finding the root of the mesh graph.
 **  The root is the tetrahedron that the
 **  probe resides on one of its faces.
 *****************************************/
Tetrahedron *FindRootofGraph(Tetrahedron *tetrahedrons, Simulation **simulations) {

	int i, j;
	bool probe_flag = true;
	double distances_from_faces[4];
	for (i = 0; i < (*simulations)->num_tetrahedrons; i++) {
		probe_flag = true;
		for (j = 0; j < 4; j++) {
			distances_from_faces[j] =
			    DotProduct((*simulations)->probe_x,
			               (*simulations)->probe_y,
			               (*simulations)->probe_z,
			               tetrahedrons[i].faces[j]->nx,
			               tetrahedrons[i].faces[j]->ny,
			               tetrahedrons[i].faces[j]->nz) +
			    tetrahedrons[i].faces[j]->d;
			if (distances_from_faces[j] < 0) {
				probe_flag = false;
				break;
			}
		}
		if (probe_flag) {
			for (j = 0; j < 4; j++)
				if (distances_from_faces[j] == 0)
					if (tetrahedrons[i].adjacent_tetrahedrons[j] == NULL)
						return &tetrahedrons[i];
					else {
						fprintf(stderr, "The probe is not placed on the surface!!!\n");
						return NULL;
					}
		}
	}

	fprintf(stderr, "The probe is outside of the medium!!!\n");
	return NULL;
}

/*************************************************
 **  Serializing the mesh graph for using in GPU
 *************************************************/
void SerializeGraph(Tetrahedron *root_tetrahedron,
                    unsigned int number_of_vertices,
                    unsigned int number_of_faces,
                    unsigned int number_of_tetrahedrons,
                    Vertex *d_vertices,
                    TriangleFaces *d_faces,
                    TetrahedronGPU *d_root) {

	int i, j;
	int root_index = root_tetrahedron[0].index;
	bool seen_faces[number_of_faces];
	long double distance;
	double mid_point_x = 0.0, mid_point_y = 0.0, mid_point_z = 0.0;


	for (i = 0; i < number_of_faces; i++)
		seen_faces[i] = false;

	for (i = 0; i < number_of_tetrahedrons; i++) {
		d_root[i].index = root_tetrahedron[i - root_index].index;
		d_root[i].region = root_tetrahedron[i - root_index].region;
		for (j = 0; j < 4; j++) {
			d_vertices[root_tetrahedron[i - root_index].vertices[j]->index].index =
			    root_tetrahedron[i - root_index].vertices[j]->index;
			d_vertices[root_tetrahedron[i - root_index].vertices[j]->index].x =
			    root_tetrahedron[i - root_index].vertices[j]->x;
			d_vertices[root_tetrahedron[i - root_index].vertices[j]->index].y =
			    root_tetrahedron[i - root_index].vertices[j]->y;
			d_vertices[root_tetrahedron[i - root_index].vertices[j]->index].z =
			    root_tetrahedron[i - root_index].vertices[j]->z;
		}

		mid_point_x = mid_point_y = mid_point_z = 0.0;
		for (j = 0; j < 4; j++) {
			mid_point_x += d_vertices[root_tetrahedron[i - root_index].vertices[j]->index].x;
			mid_point_y += d_vertices[root_tetrahedron[i - root_index].vertices[j]->index].y;
			mid_point_z += d_vertices[root_tetrahedron[i - root_index].vertices[j]->index].z;
			if (j != 0) {
				mid_point_x /= 2.0;
				mid_point_y /= 2.0;
				mid_point_z /= 2.0;
			}
		}

		for (j = 0; j < 4; j++) {
			d_root[i].faces[j] =
			    d_faces[root_tetrahedron[i - root_index].faces[j]->index].index =
			        root_tetrahedron [i - root_index].faces[j]->index;
			if (!seen_faces[d_root[i].faces[j]]) {
				d_faces[root_tetrahedron[i - root_index].faces[j]->index].d =
				    root_tetrahedron[i - root_index].faces[j]->d;
				d_faces[root_tetrahedron[i - root_index].faces[j]->index].nx =
				    root_tetrahedron[i - root_index].faces[j]->nx;
				d_faces[root_tetrahedron[i - root_index].faces[j]->index].ny =
				    root_tetrahedron[i - root_index].faces[j]->ny;
				d_faces[root_tetrahedron[i - root_index].faces[j]->index].nz =
				    root_tetrahedron[i - root_index].faces[j]->nz;
				seen_faces[d_root[i].faces[j]] = true;
			}

			distance = (d_faces[root_tetrahedron[i - root_index].faces[j]->index].nx) * mid_point_x +
			           (d_faces[root_tetrahedron[i - root_index].faces[j]->index].ny) * mid_point_y +
			           (d_faces[root_tetrahedron[i - root_index].faces[j]->index].nz) * mid_point_z +
			           d_faces[root_tetrahedron[i - root_index].faces[j]->index].d;
			if (distance < 0)
				d_root[i].signs[j] = -1;
			else
				d_root[i].signs[j] = 1;

		}

		for (j = 0; j < 4; j++) {
			if (root_tetrahedron[i - root_index].adjacent_tetrahedrons[j] == NULL)
				d_root[i].adjacent_tetrahedrons[j] = -1; // If not adjacent tetrahedron found
			else
				d_root[i].adjacent_tetrahedrons[j] = root_tetrahedron[i - root_index].adjacent_tetrahedrons[j]->index;
		}
	}
}
