set(CMAKE_C_COMPILER "aarch64-linux-gnu-gcc")
set(CMAKE_CXX_COMPILER "aarch64-linux-gnu-gcc")

project(executorch_tranformer LANGUAGES CXX)

cmake_minimum_required(VERSION 3.5)

add_subdirectory("executorch")

add_executable(gpt2_cpu gpt2_cpu.cpp)

target_link_libraries(
  gpt2_cpu
  PRIVATE executorch
          extension_module_static
          extension_tensor
          optimized_native_cpu_ops_lib
          xnnpack_backend)