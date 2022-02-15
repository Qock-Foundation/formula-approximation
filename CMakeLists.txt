cmake_minimum_required(VERSION 3.21)
project(FormulaApproximation)

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_CXX_FLAGS_ASAN "-g -fsanitize=address,undefined")

add_executable(FormulaApproximation main.cpp)
