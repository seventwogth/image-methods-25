#pragma once

#include <array>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include <embree3/rtcore.h>
#include <embree3/rtcore_ray.h>

#include "math.hpp"

struct Material {
    std::string name;
    Vec3 diffuseColor;
    Vec3 specularColor;
    double kd = 0.0;
    double ks = 0.0;
    double shininess = 1.0;
};

struct PointLight {
    std::string name;
    Vec3 position;
    Vec3 intensity;
};

struct TriangleMesh {
    std::string name;
    Material material;
    std::vector<Vec3> vertices;
    std::vector<std::array<std::uint32_t, 3>> indices;

    std::size_t triangleCount() const;
    Vec3 triangleNormal(std::size_t triangleIndex) const;
};

struct Camera {
    Vec3 position;
    Vec3 target;
    Vec3 up;
    double fovYDegrees = 45.0;
    int width = 640;
    int height = 640;

    Ray makeRay(double pixelX, double pixelY) const;
};

struct SceneData {
    Camera camera;
    std::vector<PointLight> lights;
    std::vector<TriangleMesh> meshes;
    Vec3 backgroundColor;
};

class EmbreeScene {
public:
    struct HitRecord {
        bool hit = false;
        double distance = 0.0;
        Vec3 position;
        Vec3 normal;
        const Material* material = nullptr;
        std::uint32_t geomID = RTC_INVALID_GEOMETRY_ID;
        std::uint32_t primID = RTC_INVALID_GEOMETRY_ID;
    };

    explicit EmbreeScene(const SceneData& data);
    ~EmbreeScene();

    EmbreeScene(const EmbreeScene&) = delete;
    EmbreeScene& operator=(const EmbreeScene&) = delete;

    EmbreeScene(EmbreeScene&& other) noexcept;
    EmbreeScene& operator=(EmbreeScene&& other) noexcept;

    HitRecord intersect(
        const Ray& ray,
        double tMin = 1e-4,
        double tMax = std::numeric_limits<double>::infinity()
    ) const;

    bool isOccluded(const Ray& ray, double tMin, double tMax) const;

    std::size_t triangleCount() const;

private:
    struct GeometryBinding {
        const TriangleMesh* mesh = nullptr;
    };

    RTCDevice device_ = nullptr;
    RTCScene scene_ = nullptr;
    const SceneData* data_ = nullptr;
    std::vector<GeometryBinding> geometryBindings_;

    void release();
};

SceneData createDefaultScene();
