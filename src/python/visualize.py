from __future__ import absolute_import, division, print_function, unicode_literals

"""
    This is the visualization module, using Mayavi and tvtk to visualize
    tetrahedron mesh of the OCTMPS mesh file. You can visualize two cases:
    1. Full tetrahedron mesh
    2. Surface mesh (triangular mesh) of the surfaces of the regions

"""

import numpy as np
from mayavi import mlab
from mayavi.sources.array_source import ArraySource
from mayavi.sources.vtk_data_source import VTKDataSource
from tvtk.api import tvtk
from tvtk.misc import write_data
from tvtk.tools.mlab import make_triangle_polydata

from src.python.octmps_output import parse_OCTMPS_output_file


def get_surface_structure(tetrahedrons):
    """
    Visualize the surface triangular mesh of the regions of the tetrahedron mesh
    :param tetrahedrons: list of tetrahedrons (of type Tetrahedron)
    :return: data for mayavi data source
    """
    faces_to_tetrahedrons_dict = {}
    regions = set()

    for tetrahedron in tetrahedrons:
        regions.add(tetrahedron.region)
        for i in range(4):
            try:
                faces_to_tetrahedrons_dict[tetrahedron.get_face(i)].append(tetrahedron.region)
            except KeyError:
                faces_to_tetrahedrons_dict[tetrahedron.get_face(i)] = [tetrahedron.region]

    surface_faces = filter(lambda x: len(faces_to_tetrahedrons_dict[x]) == 1 or faces_to_tetrahedrons_dict[x][0] !=
                                                                                faces_to_tetrahedrons_dict[x][1],
                           faces_to_tetrahedrons_dict.keys())

    vertices = list()
    trimesh = list()
    index = 0
    for face in surface_faces:
        tri = list()
        for i in range(3):
            v = face.get_vertex(i)
            vertices.append([v.x, v.y, v.z])
            tri.append(index)
            index += 1
        trimesh.append(tri)

    return make_triangle_polydata(triangles=trimesh, points=vertices)


def get_complete_structure(tetrahedrons):
    """
    Visualize the complete tetrahedron mesh of all regions of the tetrahedron mesh
    :param tetrahedrons: list of tetrahedrons (of type Tetrahedron)
    :return: data for mayavi data source
    """
    vertices = list()
    tets = list()

    index = 0
    for tetrahedron in tetrahedrons:
        tet_vertices_indices = []
        for i in range(4):
            v = tetrahedron.get_vertex(i)
            vertices.append([v.x, v.y, v.z])
            tet_vertices_indices.append(index)
            index += 1

        tets.append(tet_vertices_indices)

    vertices = np.array(vertices)
    tets = np.array(tets)
    tet_type = tvtk.Tetra().cell_type
    unstructured_grid = tvtk.UnstructuredGrid(points=vertices)
    unstructured_grid.set_cells(tet_type, tets)
    return unstructured_grid


def make_data(octmps_output_file='ClassI_ScattFilt.out'):
    """Creates some simple array data of the given dimensions to test
    with."""

    reflectance_grid, reflectances, x_positions, z_positions = parse_OCTMPS_output_file(file_name=octmps_output_file)
    return reflectance_grid, x_positions, z_positions


@mlab.show
def visualize(tetrahedrons,
              save=False,
              file_name='tetrahedron_mesh.vtu',
              just_surface=True,
              octmps_output_file='ClassI_ScattFilt.out'):
    if just_surface:
        data = get_surface_structure(tetrahedrons)
    else:
        data = get_complete_structure(tetrahedrons)

    # Saving as VTU file
    if save:
        write_data(data, file_name=file_name)

    mlab.figure(figure='OCTMPS', fgcolor=(1, 1, 1),
                bgcolor=(0.5, 0.5, 0.5))

    src = VTKDataSource(data=data)
    surf = mlab.pipeline.surface(src, opacity=0.01)
    mlab.axes()
    mlab.pipeline.surface(mlab.pipeline.extract_edges(surf),
                          color=(1, 1, 1), line_width=0.0)

    if octmps_output_file:
        # Make the data and add it to the pipeline.
        data, x_positions, z_positions = make_data(octmps_output_file=octmps_output_file)
        data = np.array([data]).swapaxes(1, 0)
        src = ArraySource(transpose_input_array=True)
        src.scalar_data = data
        src.spacing = (x_positions[1] - x_positions[0], 0., z_positions[1] - z_positions[0])
        src.origin = (x_positions[0], 0.0, z_positions[0])
        mlab.pipeline.surface(src, colormap='jet')
        mlab.colorbar(orientation='vertical', title='Reflectance')
        mlab.show()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Massively Parallel Simulator of Optical Coherence Tomography (OCTMPS)')

    parser.add_argument('--input-mesh-file', help='Input .mesh file',
                        dest='input_mesh_file',
                        type=str, required=True)

    parser.add_argument('--just-surface', help='Visualize just the surface',
                        dest='just_surface',
                        type=bool, required=False, default=True)

    parser.add_argument('--octmps-output-file', help='OCTMPS output file',
                        dest='octmps_output_file',
                        type=str, required=False, default=None)

    parser.add_argument('--save-as-vtk-file', help='Save Output as VTK file',
                        dest='vtk_file',
                        type=str, required=False, default=None)

    args = parser.parse_args()

    import sys

    sys.argv = sys.argv[:1]

    from src.python.mesh import load_mesh

    tetrahedrons, num_tetrahedrons, num_vertices, num_faces = load_mesh(args.input_mesh_file)

    visualize(tetrahedrons=tetrahedrons,
              save=False,
              file_name=args.vtk_file,
              just_surface=args.just_surface,
              octmps_output_file=args.octmps_output_file)
