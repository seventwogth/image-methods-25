#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "math.hpp"

struct ObjMesh {
    std::string name;
    std::vector<Vec3> vertices;
    std::vector<std::array<std::uint32_t, 3>> indices;
};

std::vector<ObjMesh> loadObj(const std::filesystem::path& path);
