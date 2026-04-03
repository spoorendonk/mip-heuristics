cmake_minimum_required(VERSION 3.25)
project(mip-heuristics LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(cmake/FetchHiGHS.cmake)

add_library(mip_heuristics OBJECT
    src/fpr.cpp
    src/fpr_core.cpp
    src/fpr_lp.cpp
    src/fpr_strategies.cpp
    src/local_mip.cpp
    src/prop_engine.cpp
    src/repair_search.cpp
    src/thompson_sampler.cpp
    src/solution_pool.cpp
    src/portfolio.cpp
    src/scylla.cpp
    src/walksat.cpp
)
target_include_directories(mip_heuristics PRIVATE ${highs_SOURCE_DIR}/highs ${highs_BINARY_DIR} src)

target_sources(highs PRIVATE $<TARGET_OBJECTS:mip_heuristics>)
target_include_directories(highs PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

# --- Tests ---
set(BUILD_TESTING ON CACHE BOOL "" FORCE)
include(FetchContent)
FetchContent_Declare(Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.7.1
)
FetchContent_MakeAvailable(Catch2)
list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(CTest)
include(Catch)

add_executable(mip_heuristics_tests tests/test_basic.cpp)
target_include_directories(mip_heuristics_tests PRIVATE ${highs_SOURCE_DIR}/highs ${highs_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(mip_heuristics_tests PRIVATE highs Catch2::Catch2WithMain)
target_compile_definitions(mip_heuristics_tests PRIVATE
    "INSTANCES_DIR=\"${highs_SOURCE_DIR}/check/instances\"")
catch_discover_tests(mip_heuristics_tests)
