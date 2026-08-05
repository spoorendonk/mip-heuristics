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
            "-DCMAKE_CUDA_COMPILER=/path/to/nvcc).\n"
            "Note: CUDA_HOME must be exported as well — see the next check.")
    endif()

    # HiGHS's FindCUDAConf.cmake (reached via CUPDLP_FIND_CUDA below) derives
    # `CMAKE_CUDA_PATH` from $CUDA_HOME and uses it for the cudart/cublas/
    # cusparse `find_library` HINTS, for HiGHS's CUDA include directory, and
    # for a plain `set(CMAKE_CUDA_COMPILER "$ENV{CUDA_HOME}/bin/nvcc")` that
    # lands in the generated build rules.  With CUDA_HOME unset those all
    # degrade to "/..." paths and fail confusingly — at configure time in the
    # REQUIRED find_library calls, or later at build time — even when nvcc is
    # on PATH.  So demand it up front with a message that names the fix.
    if(NOT DEFINED ENV{CUDA_HOME})
        message(FATAL_ERROR
            "MIP_HEURISTICS_CUDA=ON requires the CUDA_HOME environment variable "
            "(HiGHS's FindCUDAConf.cmake derives CMAKE_CUDA_PATH from it).\n"
            "Set it to your toolkit root, e.g.: export CUDA_HOME=/usr/local/cuda")
    endif()
    if(NOT EXISTS "$ENV{CUDA_HOME}/bin/nvcc")
        message(FATAL_ERROR
            "CUDA_HOME is set to '$ENV{CUDA_HOME}' but '$ENV{CUDA_HOME}/bin/nvcc' "
            "does not exist. Point CUDA_HOME at the toolkit root.")
    endif()

    enable_language(CUDA)
    # Pin the detected compiler in the cache.  FindCUDAConf later assigns a
    # *normal* variable of the same name from $CUDA_HOME (see above), so this
    # is the value CMake uses for anything reached before that point.
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

# Post-condition: assert on the macro the compiler actually sees.  Testing
# the `CUPDLP_GPU` variable here would be vacuous — we FORCE it into the
# cache ourselves above, HiGHS never clears it (its only `set(CUPDLP_GPU
# OFF)` is commented out), and anything HiGHS set inside its own directory
# scope would not propagate back to us.  `HConfig.h` is `configure_file`d at
# configure time with `#cmakedefine CUPDLP_CPU` / `#cmakedefine CUPDLP_GPU`,
# and that is precisely what `CupdlpWrapper.cpp` branches on to pick the
# device — so this checks the GPU-vs-CPU compile-time truth directly.
if(MIP_HEURISTICS_CUDA)
    set(_highs_config "${highs_BINARY_DIR}/HConfig.h")
    if(NOT EXISTS "${_highs_config}")
        message(FATAL_ERROR
            "MIP_HEURISTICS_CUDA=ON but HiGHS did not generate "
            "'${_highs_config}', so the GPU build cannot be verified.")
    endif()
    file(READ "${_highs_config}" _highs_config_text)
    if(NOT _highs_config_text MATCHES "#define +CUPDLP_GPU"
       OR _highs_config_text MATCHES "#define +CUPDLP_CPU")
        message(FATAL_ERROR
            "MIP_HEURISTICS_CUDA=ON but HiGHS generated a CPU-only cuPDLP "
            "configuration (see '${_highs_config}') — the resulting binary "
            "would run CPU-only PDLP. Check the CUDA toolkit installation "
            "(cudart, cublas and cusparse must all be findable under CUDA_HOME).")
    endif()
    unset(_highs_config_text)
    unset(_highs_config)
endif()
