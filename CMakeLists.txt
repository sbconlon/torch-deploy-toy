cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(example-proj)

find_package(TBB 2020.1 CONFIG)
if(NOT TBB_FOUND)
  # no version check possible when using the find module
  find_package(TBB MODULE REQUIRED)
endif()

find_package(Torch REQUIRED)

add_executable(example-app example-app.cpp)
target_link_libraries(example-app TBB::tbb "${TORCH_LIBRARIES}")
set_property(TARGET example-app PROPERTY CXX_STANDARD 14)
