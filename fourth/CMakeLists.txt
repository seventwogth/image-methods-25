cmake_minimum_required(VERSION 3.16)

project(fourth_lab_3_5 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(embree 3 CONFIG QUIET)

set(EMBREE_TARGET "")
if(embree_FOUND)
    if(TARGET embree)
        set(EMBREE_TARGET embree)
    elseif(TARGET embree3)
        set(EMBREE_TARGET embree3)
    elseif(TARGET embree::embree)
        set(EMBREE_TARGET embree::embree)
    endif()
endif()

if(NOT EMBREE_TARGET)
    find_path(
        EMBREE_INCLUDE_DIR
        NAMES embree3/rtcore.h embree3/rtcore_ray.h
        HINTS
            ENV EMBREE_DIR
            ENV EMBREE_ROOT
        PATH_SUFFIXES include
    )

    find_library(
        EMBREE_LIBRARY
        NAMES embree3 embree3_d
        HINTS
            ENV EMBREE_DIR
            ENV EMBREE_ROOT
        PATH_SUFFIXES lib lib64
    )

    if(EMBREE_INCLUDE_DIR AND EMBREE_LIBRARY)
        add_library(embree::fallback UNKNOWN IMPORTED)
        set_target_properties(
            embree::fallback
            PROPERTIES
                IMPORTED_LOCATION "${EMBREE_LIBRARY}"
                INTERFACE_INCLUDE_DIRECTORIES "${EMBREE_INCLUDE_DIR}"
        )
        set(EMBREE_TARGET embree::fallback)
    endif()
endif()

if(NOT EMBREE_TARGET)
    message(FATAL_ERROR
        "Intel Embree 3 was not found. "
        "Install Embree 3 and set CMAKE_PREFIX_PATH, EMBREE_DIR, or EMBREE_ROOT "
        "so that embree3/rtcore.h and the embree3 library are available."
    )
endif()

add_executable(
    fourth_lab
    src/main.cpp
    src/obj_loader.cpp
    src/scene.cpp
    src/renderer.cpp
    src/image.cpp
)

target_include_directories(fourth_lab PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(fourth_lab PRIVATE "${EMBREE_TARGET}")
target_compile_definitions(
    fourth_lab
    PRIVATE
        FOURTH_OUTPUT_DIR="${CMAKE_CURRENT_SOURCE_DIR}/output"
)

if(MSVC)
    target_compile_options(fourth_lab PRIVATE /W4 /permissive- /utf-8)
else()
    target_compile_options(fourth_lab PRIVATE -Wall -Wextra -Wpedantic)
endif()
