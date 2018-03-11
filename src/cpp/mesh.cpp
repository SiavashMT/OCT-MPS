#include "mesh.hpp"

#include <Python.h>
#include <iostream>

#include "boost_utils.hpp"
#include "octmps.h"
#include "mesh.h"
#include "simulation.h"

/*******************
**  Region Struct
*******************/

Region *InitRegion(double mua, double mus, double g, double n) {
    Region *rs = (Region *) malloc(sizeof(Region));
    rs->n     = n;
    rs->mua   = mua;
    rs->mus   = mus;
    rs->g     = g;
    if (mus == 0.0L) rs->mutr = DBL_MAX; //Glass region
    else rs->mutr = FP_ONE / (mus + mua);
    return rs;
}

boost::shared_ptr<Region> RegionWrappedInit(const boost::python::object& mua,
        const boost::python::object& mus,
        const boost::python::object& g,
        const boost::python::object& n) {

    double _mua = boost::python::extract<double>(mua);
    double _mus = boost::python::extract<double>(mus);
    double _g = boost::python::extract<double>(g);
    double _n = boost::python::extract<double>(n);
    return boost::shared_ptr<Region>(InitRegion(_mua, _mus, _g, _n));
}

void RegionExportToPython() {
    boost::python::class_<Region, boost::shared_ptr<Region>, boost::noncopyable>("Region", boost::python::no_init)
    .def("__init__", make_constructor(&RegionWrappedInit, boost::python::default_call_policies(),
                                      (boost::python::arg("mua") = 0.0,
                                       boost::python::arg("mus") = 0.0,
                                       boost::python::arg("g") = 0.0,
                                       boost::python::arg("n") = 0.0)))
    .def_readwrite("mua", &Region::mua)
    .def_readwrite("mus", &Region::mus)
    .def_readwrite("g", &Region::g)
    .def_readwrite("n", &Region::n);
}

/**************
**  Vertex
***************/

Vertex *InitVertex(int index, double x, double y, double z) {
    Vertex *v = (Vertex *) malloc(sizeof(Vertex));
    v->index = index;
    v->x = x;
    v->y = y;
    v->z = z;
    return v;
}

boost::shared_ptr<Vertex> VertexWrappedInit(const boost::python::object& index,
        const boost::python::object& x,
        const boost::python::object& y,
        const boost::python::object& z) {
    int _index = boost::python::extract<int>(index);
    double _x = boost::python::extract<double>(x);
    double _y = boost::python::extract<double>(y);
    double _z = boost::python::extract<double>(z);
    return boost::shared_ptr<Vertex>(InitVertex(_index, _x, _y, _z));
}

bool operator==(const Vertex &lhs, const Vertex &rhs ) {
    return (lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z);
}

void VertexExportToPython() {

    boost::python::class_<Vertex, boost::shared_ptr<Vertex>, boost::noncopyable>("Vertex", boost::python::no_init)
    .def("__init__", make_constructor(&VertexWrappedInit))
    .def_readwrite("index", &Vertex::index)
    .def_readwrite("x", &Vertex::x)
    .def_readwrite("y", &Vertex::y)
    .def_readwrite("z", &Vertex::z);
}

/******************
**  TriangleFaces
******************/

TriangleFaces *InitTriangleFaces(double nx, double ny, double nz, double d, int index, boost::shared_ptr<Vertex>* vertices) {
    TriangleFaces *tf = (TriangleFaces *) malloc(sizeof(TriangleFaces));

    tf->nx = nx; tf->ny = ny; tf->nz = nz; tf->d = d; tf->index = index;
    for (int i = 0; i < 3; i++)
        tf->vertices[i] = vertices[i].get();
}


boost::shared_ptr<TriangleFaces> TriangleFacesWrappedInit(const boost::python::object& nx,
        const boost::python::object& ny,
        const boost::python::object& nz,
        const boost::python::object& d,
        const boost::python::object& index,
        boost::python::list list_of_vertices) {
    double _nx = boost::python::extract<double>(nx);
    double _ny = boost::python::extract<double>(ny);
    double _nz = boost::python::extract<double>(nz);
    double _d = boost::python::extract<double>(d);
    int _index = boost::python::extract<int>(index);
    boost::shared_ptr<Vertex> *v = unwrap_python_list<boost::shared_ptr<Vertex> >(list_of_vertices);
    return boost::shared_ptr<TriangleFaces>(InitTriangleFaces(_nx, _ny, _nz, _d, _index, v));
}

template <class T>
Vertex& get_vertex(T& p, size_t index) {
    return *p.vertices[index];
}

extern "C" long ConvertToTupleOfSortedTuples(const TriangleFaces& triangle_face, PyObject *tuple) {

    /*
    ** 1. Create a list of tuples of the (x, y, z)s
    ** 2. Sort the list
    ** 3. Convert the list to Tuple
    */

    PyObject *list = PyList_New((Py_ssize_t) 3);
    PyObject **tuples = (PyObject **) malloc(sizeof(PyObject *) * 3);
    for (int i = 0; i < 3; i++) {
        tuples[i] = PyTuple_New(3);
        PyTuple_SET_ITEM(tuples[i], (Py_ssize_t)0, PyFloat_FromDouble(triangle_face.vertices[i]->x));
        PyTuple_SET_ITEM(tuples[i], (Py_ssize_t)1, PyFloat_FromDouble(triangle_face.vertices[i]->y));
        PyTuple_SET_ITEM(tuples[i], (Py_ssize_t)2, PyFloat_FromDouble(triangle_face.vertices[i]->z));
        PyList_SET_ITEM(list, (Py_ssize_t)i, tuples[i]);
    }

    if (PyList_Sort(list) != 0) {
        PyErr_SetString(PyExc_Exception, "cannot sort the list to hash!!!");
        PyErr_Print();
    }


    for (int i = 0; i < 3; i++)
        PyTuple_SET_ITEM(tuple, (Py_ssize_t)i, PyList_GET_ITEM(list, (Py_ssize_t)i));

    free(tuples);
}

extern "C" long hash(const TriangleFaces& triangle_face) {
    // Return the hash of the Tuple

    PyObject *tuple = PyTuple_New(3);

    ConvertToTupleOfSortedTuples(triangle_face, tuple);

    long hash_value = PyObject_Hash(tuple);

    return hash_value;
}

extern "C" PyObject* equal(const TriangleFaces &self, const TriangleFaces &other) {

    PyObject *self_tuple = PyTuple_New(3);
    PyObject *other_tuple = PyTuple_New(3);
    ConvertToTupleOfSortedTuples(self, self_tuple);
    ConvertToTupleOfSortedTuples(other, other_tuple);
    int cmp_result = PyObject_RichCompareBool(self_tuple, other_tuple, Py_EQ);
    if (cmp_result == -1)
        return NULL;
    else if (cmp_result == 1)
        Py_RETURN_TRUE;
    else
        Py_RETURN_FALSE;
}

TriangleFaces& GetFace(Tetrahedron& tetrahedron, size_t index) {
    return *tetrahedron.faces[index];
}

void TriangleFacesExportToPython() {
    boost::python::class_<TriangleFaces, boost::shared_ptr<TriangleFaces>, boost::noncopyable>("TriangleFaces", boost::python::no_init)
    .def("__init__", boost::python::make_constructor(&TriangleFacesWrappedInit))
    .def_readwrite("index", &TriangleFaces::index)
    .def_readwrite("nx", &TriangleFaces::nx)
    .def_readwrite("ny", &TriangleFaces::ny)
    .def_readwrite("nz", &TriangleFaces::nz)
    .def_readwrite("d", &TriangleFaces::d)
    .def("get_vertex", get_vertex<TriangleFaces>, boost::python::return_value_policy<boost::python::reference_existing_object>())
    .def("__hash__", hash)
    .def("__eq__", equal);
}

/**********************
**  Tetrahedron
**********************/

Tetrahedron* InitTetrahedron(int index,
                             int region,
                             boost::shared_ptr<TriangleFaces>* faces,
                             boost::shared_ptr<Vertex>* vertices) {
    Tetrahedron *ts = (Tetrahedron *) malloc(sizeof(Tetrahedron));
    ts->index = index; ts->region = region;
    for (int i = 0; i < 4; i++) {
        ts->faces[i] = faces[i].get();
        ts->vertices[i] = vertices[i].get();
        ts->adjacent_tetrahedrons[i] = NULL;
    }
}

boost::shared_ptr<Tetrahedron> TetrahedronWrappedInit(const boost::python::object& index,
        const boost::python::object& region,
        boost::python::list list_of_faces,
        boost::python::list list_of_vertices) {
    int _index = boost::python::extract<int>(index);
    int _region = boost::python::extract<int>(region);
    boost::shared_ptr<Vertex>* v = unwrap_python_list <boost::shared_ptr<Vertex> >(list_of_vertices);
    boost::shared_ptr<TriangleFaces>* tf = unwrap_python_list<boost::shared_ptr<TriangleFaces> >(list_of_faces);
    return boost::shared_ptr<Tetrahedron>(InitTetrahedron(_index, _region, tf, v));
}

void SetAdjacentTetrahedron(const boost::python::object& self,
                            const boost::python::object& adjacent_tetrahedron,
                            const boost::python::object& index) {
    Tetrahedron *self_ = boost::python::extract<Tetrahedron *> (self);
    Tetrahedron *adjacent_tetrahedron_ = boost::python::extract<Tetrahedron *>(adjacent_tetrahedron);
    int index_ = boost::python::extract<int>(index);
    self_->adjacent_tetrahedrons[index_] = adjacent_tetrahedron_;
}

void TetrahedronExportToPython() {
    boost::python::class_<Tetrahedron, boost::shared_ptr<Tetrahedron>, boost::noncopyable>("Tetrahedron")
    .def("__init__", make_constructor(&TetrahedronWrappedInit))
    .def_readwrite("index", &Tetrahedron::index)
    .def_readwrite("region", &Tetrahedron::region)
    .def("get_vertex", get_vertex<Tetrahedron>, boost::python::return_value_policy<boost::python::reference_existing_object>())
    .def("get_face", GetFace, boost::python::return_value_policy<boost::python::reference_existing_object>())
    .def("set_adjacent_tetrahedron", SetAdjacentTetrahedron);
}

void InitMesh() {
    boost::python::object mesh(boost::python::handle<>(boost::python::borrowed(PyImport_AddModule("octmps.mesh"))));

    boost::python::scope().attr("mesh") = mesh;

    boost::python::scope io_scope = mesh;

    RegionExportToPython();
    VertexExportToPython();
    TriangleFacesExportToPython();
    TetrahedronExportToPython();
}

