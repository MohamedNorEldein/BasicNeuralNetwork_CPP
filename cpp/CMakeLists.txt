cmake_minimum_required(VERSION 3.5)

project(MyProject)

# Set source files
set(SOURCES
    source/Stat.cpp
    source/CAPI.cpp
    source/NeuralNetworks.cpp
    )

# Add executable target
#add_executable(MyProject ${SOURCES})

# Add dll target
add_library(MyProject SHARED ${SOURCES})


