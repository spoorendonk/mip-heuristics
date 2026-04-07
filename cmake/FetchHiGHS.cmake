include(FetchContent)

# Disable HiGHS components we don't need
set(HIGHS_NO_DEFAULT_THREADS ON CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

# Optional CUDA/GPU acceleration for cuPDLP-C (used by PDLP solver in Scylla)
option(MIP_HEURISTICS_CUDA "Enable CUDA GPU acceleration for PDLP solver" OFF)
if(MIP_HEURISTICS_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        # Forward detected compiler path so HiGHS's FindCUDAConf picks it up
        set(CMAKE_CUDA_COMPILER "${CMAKE_CUDA_COMPILER}" CACHE FILEPATH "" FORCE)
        set(CUPDLP_GPU ON CACHE BOOL "" FORCE)
        set(CUPDLP_FIND_CUDA ON CACHE BOOL "" FORCE)
        message(STATUS "MIP_HEURISTICS_CUDA: enabled (CUDA compiler: ${CMAKE_CUDA_COMPILER})")
    else()
        message(WARNING "MIP_HEURISTICS_CUDA is ON but no CUDA compiler was found — falling back to CPU-only PDLP")
    endif()
endif()

FetchContent_Declare(highs
    GIT_REPOSITORY https://github.com/ERGO-Code/HiGHS.git
    GIT_TAG        v1.13.1
    PATCH_COMMAND ${CMAKE_COMMAND}
        -DPATCH_DIR=${CMAKE_CURRENT_SOURCE_DIR}/third_party/highs_patch
        -DSOURCE_DIR=<SOURCE_DIR>
        -P ${CMAKE_CURRENT_SOURCE_DIR}/third_party/highs_patch/apply_patch.cmake
)

FetchContent_MakeAvailable(highs)
