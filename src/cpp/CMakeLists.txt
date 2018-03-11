# CUDA
find_package(CUDA QUIET REQUIRED)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
set(LINK_FLAGS ${LINK_FLAGS} "-Wl,-whole-archive,--no-undefined,-z,defs")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)


message(STATUS "CUDA Found ${CUDA_FOUND}")

set(
        CUDA_NVCC_FLAGS
        ${CUDA_NVCC_FLAGS};
        -O3 -gencode arch=compute_35,code=sm_35 -rdc=true
)

cuda_include_directories(/usr/local/cuda/include/
        /usr/local/cuda/samples/common/inc/)

cuda_add_library(cuda_utils cuda_utils.cpp)

option(USE_PYTHON_VERSION "Use specific python version to build OCTMPS." "")

# Make sure libboost_python3.so is a soft link to libboost_python-py3x.so
# cd /usr/lib/x86_64-linux-gnu/
# sudo ln -s libboost_python-py3x.so libboost_python3.so

# Boost Part
if(USE_PYTHON_VERSION)
    find_package(PythonInterp ${USE_PYTHON_VERSION} REQUIRED)
    find_package(PythonLibs ${USE_PYTHON_VERSION} REQUIRED)
else()
    find_package(PythonInterp REQUIRED)
    find_package(PythonLibs REQUIRED)
endif()

if (PYTHON_VERSION_MAJOR EQUAL 3)
   find_package(Boost COMPONENTS python3 REQUIRED)
else()
   find_package(Boost COMPONENTS python REQUIRED)
endif()



include_directories(${Boost_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS})
include_directories(/usr/local/cuda/include/
        /usr/local/cuda/samples/common/inc/)

link_libraries(${Boost_LIBRARIES} ${PYTHON_LIBRARIES} octmps_kernel octmps_core cuda_utils) # Deprecated but so convenient!

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_HOME_DIRECTORY}/lib)

file(GLOB SOURCES "*.cpp")

python_add_module(octmps SHARED ${SOURCES})

target_link_libraries(octmps
        octmps_core
        octmps_kernel
        cuda_utils
        ${Boost_LIBRARIES}
        ${CUDA_LIBRARIES}
        )

install(TARGETS ${PROJECT_NAME}
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib/static
        COMPONENT libraries)
