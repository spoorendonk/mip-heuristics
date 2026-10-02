include(FetchContent)

# Disable HiGHS components we don't need.  Normal variables, not forced
# cache entries: HiGHS's option() calls honour them (CMP0077; HiGHS requires
# CMake 3.15), and they stay in this directory scope, so a parent project
# that pulls this one in keeps its own BUILD_TESTING and BUILD_SHARED_LIBS.
set(BUILD_TESTING OFF)
set(BUILD_EXAMPLES OFF)
set(BUILD_SHARED_LIBS OFF)

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
    # Normal variables, like the BUILD_* switches above and for the same
    # reason: HiGHS's option() calls honour them and a parent's cache is left
    # alone.
    set(CUPDLP_GPU ON)
    set(CUPDLP_FIND_CUDA ON)
    message(STATUS "MIP_HEURISTICS_CUDA: enabled (CUDA compiler: ${CMAKE_CUDA_COMPILER})")
endif()

# FetchContent records the patch step as a *command line* and re-runs it
# only when that line changes — editing apply_patch.cmake does not
# invalidate the stamp.  Without this, every existing build tree silently
# keeps whatever option set it was patched with, and the PATCH_VERSION
# guard inside the script never gets a chance to run: exactly epic #88's
# coupling B ("every existing HiGHS build tree contains the *old* option
# set"), which the guard exists to catch.  Feeding the script's hash into
# the command line makes any edit re-run the patch, so a stale tree hits
# the guard and gets told to clean instead of compiling against a header
# that no longer matches src/.  The variable is deliberately unused by
# the script — its only job is to be part of the stamped command.
file(SHA256 ${CMAKE_CURRENT_SOURCE_DIR}/third_party/highs_patch/apply_patch.cmake
     MIP_HEURISTICS_PATCH_SHA)

# A parent project's own HiGHS patches, layered on the same tree in the same
# PATCH_COMMAND, after ours: each script runs as `cmake -DSOURCE_DIR=<tree>
# -P <script>`, in list order, and owns its own idempotency and version
# guard — this file only runs them.  Each script's hash goes into its command
# for the reason given above, so editing one re-runs the whole patch step.
# Patching the fetched tree from outside instead would race FetchContent's
# patch stamp.  Empty (the default, and always the case top-level) appends
# nothing, so the stamped command is exactly the built-in one.  A normal
# variable set by the parent before FetchContent_MakeAvailable takes
# precedence over this cache entry (CMP0126).
set(MIP_HEURISTICS_EXTRA_HIGHS_PATCHES "" CACHE STRING
    "Absolute paths of .cmake scripts run on the HiGHS tree after the built-in patch")
set(_extra_patch_commands "")
foreach(_script IN LISTS MIP_HEURISTICS_EXTRA_HIGHS_PATCHES)
    if(NOT IS_ABSOLUTE "${_script}" OR NOT EXISTS "${_script}")
        message(FATAL_ERROR
            "MIP_HEURISTICS_EXTRA_HIGHS_PATCHES: '${_script}' is not an "
            "absolute path to an existing file.")
    endif()
    file(SHA256 "${_script}" _script_sha)
    list(APPEND _extra_patch_commands
        COMMAND ${CMAKE_COMMAND}
            -DSOURCE_DIR=<SOURCE_DIR>
            -DPATCH_SCRIPT_SHA=${_script_sha}
            -P ${_script})
endforeach()
unset(_script)
unset(_script_sha)

FetchContent_Declare(highs
    GIT_REPOSITORY https://github.com/ERGO-Code/HiGHS.git
    GIT_TAG        v1.15.1
    # Depth-1.  The full clone is ~197 MB of which ~178 MB is history we never
    # read, and the pre-push gate re-clones on every clean rebuild.  Safe with
    # a tag: the fetched ref is exactly the commit named above, and HiGHS's own
    # version banner still resolves its githash from it.
    GIT_SHALLOW    ON
    PATCH_COMMAND ${CMAKE_COMMAND}
        -DPATCH_DIR=${CMAKE_CURRENT_SOURCE_DIR}/third_party/highs_patch
        -DSOURCE_DIR=<SOURCE_DIR>
        -DPATCH_SCRIPT_SHA=${MIP_HEURISTICS_PATCH_SHA}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/third_party/highs_patch/apply_patch.cmake
        ${_extra_patch_commands}
)
unset(_extra_patch_commands)

FetchContent_MakeAvailable(highs)

# Post-condition: assert on the macro the compiler actually sees.  Testing
# the `CUPDLP_GPU` variable here would be vacuous — we set it ourselves
# above, HiGHS never clears it (its only `set(CUPDLP_GPU OFF)` is commented
# out), and anything HiGHS set inside its own directory scope would not
# propagate back to us.  `HConfig.h` is `configure_file`d at
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
