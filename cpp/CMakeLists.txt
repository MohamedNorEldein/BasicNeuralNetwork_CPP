cmake_minimum_required(VERSION 3.5)

project(MyProject)

# Set source files
set(SOURCES
    source/Stat.cpp
    )

# Add executable target
add_library(MyProject SHARED ${SOURCES})


