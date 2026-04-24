#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "image.hpp"
#include "scene.hpp"

struct RenderSettings {
    int width = 640;
    int height = 640;
    double gamma = 2.2;
    double shadowBias = 1e-3;
    std::size_t samplesPerPixel = 1;
};

struct RenderSummary {
    int width = 0;
    int height = 0;
    std::size_t triangleCount = 0;
    std::size_t lightCount = 0;
    std::size_t primaryRayCount = 0;
    std::size_t primaryHitCount = 0;
    std::size_t shadowRayCount = 0;
    std::size_t blockedShadowRayCount = 0;
    double averageLuminance = 0.0;
    double maxLuminance = 0.0;
};

struct ControlRayRecord {
    std::string name;
    int pixelX = 0;
    int pixelY = 0;
    bool hit = false;
    double distance = 0.0;
    std::uint32_t geomID = RTC_INVALID_GEOMETRY_ID;
    std::uint32_t primID = RTC_INVALID_GEOMETRY_ID;
    Vec3 hitPoint;
    Vec3 normal;
    Vec3 color;
    std::string note;
};

struct RenderOutput {
    Image image;
    RenderSummary summary;
    std::vector<ControlRayRecord> controlRays;
};

class Renderer {
public:
    explicit Renderer(RenderSettings settings);

    RenderOutput render(const SceneData& scene, const EmbreeScene& embreeScene) const;

private:
    RenderSettings settings_;

    Vec3 shade(
        const SceneData& scene,
        const EmbreeScene& embreeScene,
        const Ray& ray,
        const EmbreeScene::HitRecord& hit,
        RenderSummary& summary
    ) const;
};
