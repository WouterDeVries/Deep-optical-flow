cmake_minimum_required(VERSION 2.8)
find_package(CUDA)
find_package(PythonLibs 3.6 REQUIRED)

include_directories(
${PYTHON_INCLUDE_DIRS}
pybind11/include/
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

cuda_add_library(array_op SHARED
  array_op.cpp
  array_op.cu)

target_link_libraries(array_op
  ${PYTHON_LIBRARIES})

set_target_properties(array_op PROPERTIES PREFIX "")
