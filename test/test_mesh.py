import unittest
from lib import octmps
from src.python.mesh import make_triangle_face
from random import random
import itertools


class TestMesh(unittest.TestCase):

    def test_triangle_face_hash(self):

        vertices = [octmps.mesh.Vertex(i, random(), random(), random()) for i in range(3)]

        hash_sorted_value = hash(tuple(sorted([tuple([v.x, v.y, v.z]) for v in vertices])))

        for different_order_vertices in itertools.permutations(vertices):
            triangle_face = make_triangle_face(list(different_order_vertices), index=0)
            self.assertEqual(hash_sorted_value, hash(triangle_face))

    def test_triangle_face_equal(self):

        vertices = [octmps.mesh.Vertex(i, random(), random(), random()) for i in range(3)]

        triangle_faces = list()
        for different_order_vertices in itertools.permutations(vertices):
            triangle_faces.append(make_triangle_face(list(different_order_vertices), index=0))

        self.assertTrue(all(x == triangle_faces[0] for x in triangle_faces))

        _ = [octmps.mesh.Vertex(i, random(), random(), random()) for i in range(3)]
        different_item = make_triangle_face(list(different_order_vertices), index=0)
        self.assertTrue(all(x != different_item for x in triangle_faces))

    # Mutable Tests are to make sure that if python object is removed the back-end C++ object is not
    # deleted!!!

    def test_mutable(self):
        vertices = [octmps.mesh.Vertex(i, 1.0, 1.0, 1.0) for i in range(3)]

        triangle_face = make_triangle_face(vertices=vertices, index=0)

        self.assertEqual(triangle_face.get_vertex(0).x, 1.0)

        vertices[0].x = 2.0

        self.assertEqual(triangle_face.get_vertex(0).x, 2.0)

    def test_mutable_1(self):
        vertices = [octmps.mesh.Vertex(0, 0.0, 0.0, 0.0),
                    octmps.mesh.Vertex(1, 0.0, 1.0, 0.0),
                    octmps.mesh.Vertex(2, 1.0, 0.0, 0.0)]
        triangle_face1 = make_triangle_face(vertices=vertices, index=0)

        vertices = [octmps.mesh.Vertex(0, 0.0, 0.0, 1.0),
                    octmps.mesh.Vertex(1, 0.0, 1.0, 1.0),
                    octmps.mesh.Vertex(2, 1.0, 0.0, 1.0)]
        triangle_face2 = make_triangle_face(vertices=vertices, index=0)

        # Triangle Face 1
        self.assertEqual(triangle_face1.get_vertex(0).x, 0.0)
        self.assertEqual(triangle_face1.get_vertex(0).y, 0.0)
        self.assertEqual(triangle_face1.get_vertex(0).z, 0.0)

        self.assertEqual(triangle_face1.get_vertex(1).x, 0.0)
        self.assertEqual(triangle_face1.get_vertex(1).y, 1.0)
        self.assertEqual(triangle_face1.get_vertex(1).z, 0.0)

        self.assertEqual(triangle_face1.get_vertex(2).x, 1.0)
        self.assertEqual(triangle_face1.get_vertex(2).y, 0.0)
        self.assertEqual(triangle_face1.get_vertex(2).z, 0.0)

        # Triangle Face 2
        self.assertEqual(triangle_face2.get_vertex(0).x, 0.0)
        self.assertEqual(triangle_face2.get_vertex(0).y, 0.0)
        self.assertEqual(triangle_face2.get_vertex(0).z, 1.0)

        self.assertEqual(triangle_face2.get_vertex(1).x, 0.0)
        self.assertEqual(triangle_face2.get_vertex(1).y, 1.0)
        self.assertEqual(triangle_face2.get_vertex(1).z, 1.0)

        self.assertEqual(triangle_face2.get_vertex(2).x, 1.0)
        self.assertEqual(triangle_face2.get_vertex(2).y, 0.0)
        self.assertEqual(triangle_face2.get_vertex(2).z, 1.0)

    def test_mutable_2(self):

        t1vs = [[0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0]]

        # Tetrahedron 1
        vertices = [octmps.mesh.Vertex(i, *t1vs[i]) for i in range(4)]
        triangle_faces = []
        for i, vs in enumerate(itertools.combinations(vertices, 3)):
            triangle_faces.append(make_triangle_face(vertices=list(vs), index=i))
        tetrahedron1 = octmps.mesh.Tetrahedron(0, 0, triangle_faces, vertices)

        # Tetrahedron 2
        t2vs = [[0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0]]

        vertices = [octmps.mesh.Vertex(i, *t2vs[i]) for i in range(4)]
        triangle_faces = []
        for i, vs in enumerate(itertools.combinations(vertices, 3)):
            triangle_faces.append(make_triangle_face(vertices=list(vs), index=i))
        tetrahedron2 = octmps.mesh.Tetrahedron(1, 0, triangle_faces, vertices)

        # Triangle Face 1
        for i in range(4):
            tv = tetrahedron1.get_vertex(i)
            self.assertListEqual([tv.x, tv.y, tv.z], t1vs[i])

        for i, vs in enumerate(itertools.combinations(t1vs, 3)):
            tf = tetrahedron1.get_face(i)
            self.assertListEqual([[tf.get_vertex(i).x,
                                   tf.get_vertex(i).y,
                                   tf.get_vertex(i).z] for i in range(3)], list(vs))

        # Triangle Face 2
        for i in range(4):
            tv = tetrahedron2.get_vertex(i)
            self.assertListEqual([tv.x, tv.y, tv.z], t2vs[i])

        for i, vs in enumerate(itertools.combinations(t2vs, 3)):
            tf = tetrahedron2.get_face(i)
            self.assertListEqual([[tf.get_vertex(i).x,
                                   tf.get_vertex(i).y,
                                   tf.get_vertex(i).z] for i in range(3)], list(vs))

        modified_t2vs = [[0.0, 1.0, 12.0], [1.0, 0.0, 12.0], [0.0, 0.0, 13.0], [0.0, 0.0, 11.0]]

        vertices[0].z = 12
        vertices[1].z = 12
        vertices[2].z = 13
        vertices[3].z = 11

        for i in range(4):
            tv = tetrahedron2.get_vertex(i)
            self.assertListEqual([tv.x, tv.y, tv.z], modified_t2vs[i])

        for i, vs in enumerate(itertools.combinations(modified_t2vs, 3)):
            tf = tetrahedron2.get_face(i)
            self.assertListEqual([[tf.get_vertex(i).x,
                                   tf.get_vertex(i).y,
                                   tf.get_vertex(i).z] for i in range(3)], list(vs))


if __name__ == "__main__":
    unittest.main()
