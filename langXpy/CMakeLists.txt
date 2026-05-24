cmake_minimum_required(VERSION 3.14)
project(langXpy)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# =============================================================================
# Dependencies
# =============================================================================

include(FetchContent)

find_package(pybind11 QUIET)
if(NOT pybind11_FOUND)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11
        GIT_TAG        v2.13.6
    )
    FetchContent_MakeAvailable(pybind11)
endif()

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# =============================================================================
# langx_core Python extension
# =============================================================================

set(LANGX_HEADER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../CMake_LangX" CACHE PATH "Path to langX.h / LxRAG.h")
set(LANGX_LIB_DIR    ""                                           CACHE PATH "Path to langX.lib / langX.dll (build output)")

pybind11_add_module(langx_core "bindings.cpp")
target_include_directories(langx_core PRIVATE ${LANGX_HEADER_DIR})

if(LANGX_LIB_DIR)
    target_link_directories(langx_core PRIVATE ${LANGX_LIB_DIR})
endif()
target_link_libraries(langx_core PRIVATE langX)

add_custom_command(TARGET langx_core POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:langx_core>
        "${CMAKE_CURRENT_SOURCE_DIR}/langx/$<TARGET_FILE_NAME:langx_core>"
    COMMENT "Copying langx_core into langXpy/langx/"
)
