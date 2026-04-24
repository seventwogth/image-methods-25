#include "obj_loader.hpp"

#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

namespace {

struct ParsingMesh {
    ObjMesh mesh;
    std::unordered_map<std::uint32_t, std::uint32_t> globalToLocal;
};

std::string trim(const std::string& text) {
    std::size_t begin = 0;
    while (begin < text.size() && std::isspace(static_cast<unsigned char>(text[begin])) != 0) {
        ++begin;
    }

    std::size_t end = text.size();
    while (end > begin && std::isspace(static_cast<unsigned char>(text[end - 1])) != 0) {
        --end;
    }

    return text.substr(begin, end - begin);
}

std::uint32_t parseFaceIndexToken(const std::string& token, std::size_t lineNumber) {
    const std::size_t slashPos = token.find('/');
    const std::string indexToken = slashPos == std::string::npos ? token : token.substr(0, slashPos);
    if (indexToken.empty()) {
        throw std::runtime_error("OBJ parse error at line " + std::to_string(lineNumber) + ": face index is missing.");
    }

    int parsedIndex = 0;
    std::istringstream stream(indexToken);
    stream >> parsedIndex;
    if (!stream || !stream.eof() || parsedIndex <= 0) {
        throw std::runtime_error(
            "OBJ parse error at line " + std::to_string(lineNumber) + ": invalid face index '" + token + "'."
        );
    }

    return static_cast<std::uint32_t>(parsedIndex - 1);
}

void finalizeMesh(std::vector<ObjMesh>& meshes, ParsingMesh& currentMesh) {
    if (!currentMesh.mesh.indices.empty()) {
        meshes.push_back(std::move(currentMesh.mesh));
    }

    currentMesh = {};
}

} // namespace

std::vector<ObjMesh> loadObj(const std::filesystem::path& path) {
    if (!std::filesystem::exists(path)) {
        throw std::runtime_error("OBJ file was not found: " + path.string());
    }

    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("Failed to open OBJ file: " + path.string());
    }

    std::vector<Vec3> globalVertices;
    std::vector<ObjMesh> meshes;
    ParsingMesh currentMesh;
    currentMesh.mesh.name = "default";

    std::string line;
    std::size_t lineNumber = 0;
    while (std::getline(input, line)) {
        ++lineNumber;

        const std::string trimmed = trim(line);
        if (trimmed.empty() || trimmed[0] == '#') {
            continue;
        }

        std::istringstream stream(trimmed);
        std::string keyword;
        stream >> keyword;

        if (keyword == "o") {
            finalizeMesh(meshes, currentMesh);

            std::string objectName;
            std::getline(stream >> std::ws, objectName);
            currentMesh.mesh.name = objectName.empty() ? "unnamed_object" : objectName;
            continue;
        }

        if (keyword == "v") {
            double x = 0.0;
            double y = 0.0;
            double z = 0.0;
            stream >> x >> y >> z;
            if (!stream) {
                throw std::runtime_error("OBJ parse error at line " + std::to_string(lineNumber) + ": invalid vertex.");
            }

            globalVertices.emplace_back(x, y, z);
            continue;
        }

        if (keyword == "f") {
            std::string iToken;
            std::string jToken;
            std::string kToken;
            std::string extraToken;
            stream >> iToken >> jToken >> kToken;
            if (!stream || (stream >> extraToken)) {
                throw std::runtime_error(
                    "OBJ parse error at line " + std::to_string(lineNumber) + ": only triangle faces are supported."
                );
            }

            const std::array<std::uint32_t, 3> globalTriangle = {
                parseFaceIndexToken(iToken, lineNumber),
                parseFaceIndexToken(jToken, lineNumber),
                parseFaceIndexToken(kToken, lineNumber)
            };

            std::array<std::uint32_t, 3> localTriangle{};
            for (std::size_t vertexIndex = 0; vertexIndex < globalTriangle.size(); ++vertexIndex) {
                const std::uint32_t globalIndex = globalTriangle[vertexIndex];
                if (globalIndex >= globalVertices.size()) {
                    throw std::runtime_error(
                        "OBJ parse error at line " + std::to_string(lineNumber) + ": face index is out of range."
                    );
                }

                const auto found = currentMesh.globalToLocal.find(globalIndex);
                if (found != currentMesh.globalToLocal.end()) {
                    localTriangle[vertexIndex] = found->second;
                    continue;
                }

                const std::uint32_t localIndex = static_cast<std::uint32_t>(currentMesh.mesh.vertices.size());
                currentMesh.mesh.vertices.push_back(globalVertices[globalIndex]);
                currentMesh.globalToLocal.emplace(globalIndex, localIndex);
                localTriangle[vertexIndex] = localIndex;
            }

            currentMesh.mesh.indices.push_back(localTriangle);
            continue;
        }
    }

    finalizeMesh(meshes, currentMesh);

    if (globalVertices.empty()) {
        throw std::runtime_error("OBJ file does not contain any vertices: " + path.string());
    }

    if (meshes.empty()) {
        throw std::runtime_error("OBJ file does not contain any triangle meshes: " + path.string());
    }

    return meshes;
}
