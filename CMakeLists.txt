cmake_minimum_required(VERSION 3.20)
project(NeuSimRamulator LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(RAMULATOR_PYTHON_BINDINGS OFF CACHE BOOL "Use our incremental C ABI" FORCE)
# The pinned upstream header is rejected by AppleClang's template parser.
# Generate a one-token portability overlay; never modify the upstream checkout.
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ramulator2/src/ramulator/base/param.h" PARAM_HEADER)
string(REPLACE "config[name].as<T>()" "config[name].template as<T>()" PARAM_HEADER "${PARAM_HEADER}")
file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/compat/ramulator/base")
file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/compat/ramulator/base/param.h" CONTENT "${PARAM_HEADER}")
include_directories(BEFORE "${CMAKE_BINARY_DIR}/compat")
add_subdirectory(third_party/ramulator2 ramulator-build)
add_library(neusim_ramulator_bridge SHARED native/bridge.cpp)
target_link_libraries(neusim_ramulator_bridge PRIVATE ramulator fmt::fmt yaml-cpp::yaml-cpp)
set_target_properties(neusim_ramulator_bridge PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
