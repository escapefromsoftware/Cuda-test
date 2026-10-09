cmake_minimum_required(VERSION 3.20)

project(CrowdSimulation LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(glfw3 CONFIG REQUIRED)
find_package(glad CONFIG REQUIRED)
find_package(OpenGL REQUIRED)

add_executable(CrowdSimulation
    src/main.cpp
)

target_link_libraries(CrowdSimulation PRIVATE
    glfw
    glad::glad
    OpenGL::GL
)