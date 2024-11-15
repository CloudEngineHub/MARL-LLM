cmake_minimum_required(VERSION 3.0)

project(SwarmEnv)

IF (APPLE)
    set(CMAKE_CXX_COMPILER "/usr/local/opt/llvm/bin/clang++")
    link_directories("/usr/local/opt/llvm/lib")
    include_directories( "/usr/local/include" )
    link_directories("/usr/local/lib/")
ENDIF()

# set(CMAKE_BUILD_TYPE Debug)

file(GLOB flocking_sources src/FlockingEnv.cpp)
file(GLOB formation_sources src/FormationEnv.cpp)
file(GLOB predator_prey_sources src/PredatorPreyEnv.cpp)
file(GLOB adversarial_sources src/AdversarialEnv.cpp)
file(GLOB test_sources src/test.cpp)
set(LIB_SRC_FILES_1 ${flocking_sources})
set(LIB_SRC_FILES_2 ${formation_sources})
set(LIB_SRC_FILES_3 ${predator_prey_sources})
set(LIB_SRC_FILES_4 ${adversarial_sources})
set(LIB_SRC_TEST_FILES ${test_sources})

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Wall -std=c++11 -O3")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-variable -Wno-reorder -Wno-sign-compare -Wno-missing-braces")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -pthread -DDEBUG")

# include
include_directories(src)

# runtime library
add_library(FlockingEnv SHARED ${LIB_SRC_FILES_1}) # build shared library
add_library(FormationEnv SHARED ${LIB_SRC_FILES_2}) # build shared library
add_library(PredatorPreyEnv SHARED ${LIB_SRC_FILES_3}) # build shared library
add_library(AdversarialEnv SHARED ${LIB_SRC_FILES_4}) # build shared library
add_executable(testlib ${LIB_SRC_TEST_FILES}) # build executable file


