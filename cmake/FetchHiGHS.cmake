include(FetchContent)

# Disable HiGHS components we don't need
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

# Optional CUDA/GPU acceleration for cuPDLP-C (used by PDLP solver in Scylla)
#
# GPU vs CPU is a *compile-time* choice in HiGHS: `CupdlpWrapper.cpp` picks
# `data->device` behind `#ifdef CUPDLP_CPU`, and no runtime option can
# override it.  A configure that quietly degrades to CPU therefore produces
# a binary indistinguishable from a GPU one at the command line, which is a
# benchmarking hazard — so every failure below is fatal rather than a
# warning.  Configure without `-DMIP_HEURISTICS_CUDA=ON` to get CPU PDLP.
option(MIP_HEURISTICS_CUDA "Enable CUDA GPU acceleration for PDLP solver" OFF)
if(MIP_HEURISTICS_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(NOT CMAKE_CUDA_COMPILER)
        message(FATAL_ERROR
            "MIP_HEURISTICS_CUDA=ON but no CUDA compiler was found.\n"
            "Install the CUDA toolkit and put nvcc on PATH (or pass "
            "-DCMAKE_CUDA_COMPILER=/path/to/nvcc).")
    endif()

    # HiGHS's FindCUDAConf.cmake (reached via CUPDLP_FIND_CUDA below) does a
    # plain `set(CMAKE_CUDA_COMPILER "$ENV{CUDA_HOME}/bin/nvcc")`, which
    # shadows the cache entry we force just below.  With CUDA_HOME unset it
    # resolves to "/bin/nvcc" and fails confusingly, even when nvcc is on
    # PATH — so demand CUDA_HOME up front with a message that names the fix.
    if(NOT DEFINED ENV{CUDA_HOME})
        message(FATAL_ERROR
            "MIP_HEURISTICS_CUDA=ON requires the CUDA_HOME environment variable "
            "(HiGHS's FindCUDAConf.cmake resolves nvcc as $CUDA_HOME/bin/nvcc).\n"
            "Set it to your toolkit root, e.g.: export CUDA_HOME=/usr/local/cuda")
    endif()
    if(NOT EXISTS "$ENV{CUDA_HOME}/bin/nvcc")
        message(FATAL_ERROR
            "CUDA_HOME is set to '$ENV{CUDA_HOME}' but '$ENV{CUDA_HOME}/bin/nvcc' "
            "does not exist. Point CUDA_HOME at the toolkit root.")
    endif()

    enable_language(CUDA)
    # Forward detected compiler path so HiGHS's FindCUDAConf picks it up
    set(CMAKE_CUDA_COMPILER "${CMAKE_CUDA_COMPILER}" CACHE FILEPATH "" FORCE)
    set(CUPDLP_GPU ON CACHE BOOL "" FORCE)
    set(CUPDLP_FIND_CUDA ON CACHE BOOL "" FORCE)
    message(STATUS "MIP_HEURISTICS_CUDA: enabled (CUDA compiler: ${CMAKE_CUDA_COMPILER})")
endif()

FetchContent_Declare(highs
    GIT_REPOSITORY https://github.com/ERGO-Code/HiGHS.git
    GIT_TAG        v1.15.1
    PATCH_COMMAND ${CMAKE_COMMAND}
        -DPATCH_DIR=${CMAKE_CURRENT_SOURCE_DIR}/third_party/highs_patch
        -DSOURCE_DIR=<SOURCE_DIR>
        -P ${CMAKE_CURRENT_SOURCE_DIR}/third_party/highs_patch/apply_patch.cmake
)

FetchContent_MakeAvailable(highs)

# Post-condition: HiGHS's own CMakeLists can still flip CUPDLP_GPU off (e.g.
# an unmet CUDA dependency inside FindCUDAConf). Catch that here so the
# "silent CPU binary" failure mode cannot survive a successful configure.
if(MIP_HEURISTICS_CUDA AND NOT CUPDLP_GPU)
    message(FATAL_ERROR
        "MIP_HEURISTICS_CUDA=ON but HiGHS disabled CUPDLP_GPU during configure — "
        "the resulting binary would run CPU-only PDLP. Check the CUDA toolkit "
        "installation (cudart, cublas and cusparse must all be findable under "
        "$ENV{CUDA_HOME}).")
endif()
