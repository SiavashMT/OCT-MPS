#ifndef OCTMPS_INCLUDE_BOOST_UTILS_HPP_
#define OCTMPS_INCLUDED_BOOST_UTILS_HPP_

#include <boost/python.hpp>

template <typename T>
T* unwrap_python_list(boost::python::list python_list)
{
    std::size_t n = boost::python::len(python_list);

    T* unwrapped_pylist = new T [n];
    for (int i = 0; i < n; i++) {
        boost::python::object pyobj = python_list[i];
        unwrapped_pylist[i] = boost::python::extract<T>(python_list[i]);
    }
    return unwrapped_pylist;
}

#endif