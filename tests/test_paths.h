#pragma once

#include <string>

// Path to the HiGHS-provided `check/instances/` directory.  Only
// `test_paths.cpp` sees the `INSTANCES_DIR` define CMake injects: it holds
// the build tree's absolute path, which ccache's `base_dir` does not rewrite
// inside a `-D` value, so as a target-wide define it would make every test
// file miss the cache in each new pre-push worktree.
const std::string& instances_dir();
