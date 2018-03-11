project(octmps_core)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

file(GLOB SOURCES *.c)

set(LINK_FLAGS ${LINK_FLAGS} "-Wl,-whole-archive")
add_library(${PROJECT_NAME} SHARED ${SOURCES})
target_link_libraries(${PROJECT_NAME})
install(TARGETS ${PROJECT_NAME}
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib/static
        COMPONENT libraries)

