#ifndef OCTMPS_INCLUDE_MESH_HPP_
#define OCTMPS_INCLUDE_MESH_HPP_

#include <boost/shared_ptr.hpp>
#include <boost/functional/hash.hpp>
#include <boost/python/args.hpp>
#include <boost/python.hpp>
#include <boost/python/module.hpp>

#include "mesh.h"
#include "octmps.h"

// Regions methods
Region *InitRegion(double, double, double, double);

void RegionExportToPython();

// Vertex methods
Vertex *InitVertex(int, double, double, double);

bool operator==(const Vertex &lhs, const Vertex &rhs );

void VertexExportToPython();

// TriangleFaces methods
TriangleFaces *InitTriangleFaces(double, double, double, double, int, boost::shared_ptr<Vertex> *);

void TriangleFacesExportToPython();

// Tetrahedron methods
Tetrahedron *InitTetrahedron(int, int, boost::shared_ptr<TriangleFaces> *, boost::shared_ptr<Vertex> *);

bool operator==(const Tetrahedron &lhs, const Tetrahedron &rhs);

void TetrahedronExportToPython();

boost::shared_ptr<Vertex> VertexWrappedInit(const boost::python::object&,
                                              const boost::python::object&,
                                              const boost::python::object&,
                                              const boost::python::object&);

boost::shared_ptr<TriangleFaces> TriangleFacesWrappedInit(const boost::python::object&,
                                                            const boost::python::object&,
                                                            const boost::python::object&,
                                                            const boost::python::object&,
                                                            const boost::python::object&,
                                                            boost::python::list);

template <class T> Vertex& get_vertex(T&, size_t);

TriangleFaces& GetFace(Tetrahedron&, size_t);

extern "C" long ConvertToTupleOfSortedTuples(const TriangleFaces&, PyObject *);

extern "C" long hash(const TriangleFaces&);

extern "C" PyObject* equal(const TriangleFaces &, const TriangleFaces &);

/**********************
**  Tetrahedron
**********************/

boost::shared_ptr<Tetrahedron> TetrahedronWrappedInit(const boost::python::object&,
                                                                    const boost::python::object&,
                                                                    boost::python::list,
                                                                    boost::python::list);

void InitMesh();

#endif