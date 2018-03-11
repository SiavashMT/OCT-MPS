from __future__ import absolute_import, division, print_function, unicode_literals

"""
    Notes: For triangle faces there were two way of implementation.
    (1) Triangle faces are two triangle face per tetrahedron (if it
    shares a faces with another tetrahedron)
    (2) just one triangle face but normal to surfaces are negative of
    each other.
    We went with the first approach as it was easier to handle and code!
"""
import itertools
from lib import octmps
import numpy as np


def distance(face, np_vector):
    return face.nx * np_vector[0] + face.ny * np_vector[1] + face.nz * np_vector[2] + face.d


def make_inward_normal(tetrahedron):
    """
    This function is to make sure that normal of the four faces of the tetrahedrons are
    all pointing inward. This is done by finding a midpoint - inside the tetrahedron -
    and then finiding the distatnce from the midpoint to the face and reversing the normal
    if distance is negative.
    :param tetrahedron: Tetrahedron Struct
    :return:
    """

    convert_to_np_array = lambda v: np.array([v.x, v.y, v.z])
    np_vertices = list(map(convert_to_np_array, [tetrahedron.get_vertex(i) for i in range(4)]))
    # This is the middle point
    # midpoint = np.mean(np_vertices, axis=0)

    midpoint = np_vertices[0]
    for i in range(1, 4):
            midpoint += np_vertices[i]
            midpoint = midpoint / 2.0

    for i in range(4):
        face = tetrahedron.get_face(i)
        d = distance(face, midpoint)
        if d < 0:
            face.nx *= -1.0
            face.ny *= -1.0
            face.nz *= -1.0
            face.d *= -1.0


def make_triangle_face(vertices, index):
    difference_vector = lambda v1, v2: np.array([v1.x - v2.x, v1.y - v2.y, v1.z - v2.z])
    # Calculating the normal and the offset
    normal = np.cross(difference_vector(vertices[0], vertices[1]), difference_vector(vertices[0], vertices[2]))
    normal /= np.linalg.norm(normal)
    d = np.dot(-1 * normal, np.array([vertices[0].x, vertices[0].y, vertices[0].z]))

    return octmps.mesh.TriangleFaces(normal[0], normal[1], normal[2], d, index, vertices)


def connect_tetrahedrons(tetrahedrons):

    face_tetrahedron_dict = dict()

    for tetrahedron in tetrahedrons:
        for i in range(4):
            try:
                face_tetrahedron_dict[tetrahedron.get_face(i)].append((tetrahedron, i))
            except KeyError:
                face_tetrahedron_dict[tetrahedron.get_face(i)] = [(tetrahedron, i)]

    # Connect tetrahedrons
    for tetrahedron in tetrahedrons:
        for i in range(4):
            triangle_face = tetrahedron.get_face(i)
            adjacent_tetrahedrons_index = face_tetrahedron_dict[triangle_face]
            if len(adjacent_tetrahedrons_index) == 2:
                adjacent_tetrahedrons_index[0][0].set_adjacent_tetrahedron(adjacent_tetrahedrons_index[1][0],
                                                                           adjacent_tetrahedrons_index[0][1])
                adjacent_tetrahedrons_index[1][0].set_adjacent_tetrahedron(adjacent_tetrahedrons_index[0][0],
                                                                           adjacent_tetrahedrons_index[1][1])


def load_mesh(mesh_file_name, delimiter='\t'):
    with open(mesh_file_name, 'r') as mesh_file:
        num_vertices = int(next(mesh_file))
        num_tetrahedrons = int(next(mesh_file))
        num_faces = 0

        vertices = [octmps.mesh.Vertex(i, *map(lambda x: float(x), line.split(delimiter)))
                    for i, line in enumerate(itertools.islice(mesh_file, 0, num_vertices))]

        triangle_faces = []
        tetrahedrons = []

        for index, line in enumerate(mesh_file):
            line = [int(x) for x in line.split(delimiter)]
            ids, region = line[:-1], line[-1]
            faces_vertices_ids = itertools.combinations(ids, 3)
            tetrahedron_triangle_faces = []
            tetrahedron_vertices = [vertices[i - 1] for i in ids]
            for face_vertices_ids in faces_vertices_ids:
                v = [vertices[face_vertices_ids[i] - 1] for i in range(3)]
                triangle_face = make_triangle_face(v, index=num_faces)
                num_faces += 1
                triangle_faces.append(triangle_face)
                tetrahedron_triangle_faces.append(triangle_face)
            tetrahedrons.append(
                octmps.mesh.Tetrahedron(int(index), int(region), tetrahedron_triangle_faces, tetrahedron_vertices))

        for tetrahedron in tetrahedrons:
            make_inward_normal(tetrahedron)

        connect_tetrahedrons(tetrahedrons)

    return tetrahedrons, num_tetrahedrons, num_vertices, num_faces

