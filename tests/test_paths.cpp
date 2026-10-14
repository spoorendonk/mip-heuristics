#include "test_paths.h"

const std::string& instances_dir() {
    static const std::string kDir = INSTANCES_DIR;
    return kDir;
}
