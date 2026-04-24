#include "renderer.hpp"

#include <algorithm>
#include <array>
#include <cmath>

namespace {

struct ControlPixel {
    const char* name;
    int x;
    int y;
};

std::vector<ControlPixel> makeControlPixels(int width, int height) {
    return {
        {"center", width / 2, height / 2},
        {"upper_left", width / 4, height / 4},
        {"upper_right", (3 * width) / 4, height / 4},
        {"lower_left", width / 4, (3 * height) / 4},
        {"lower_right", (3 * width) / 4, (3 * height) / 4}
    };
}

} // namespace

Renderer::Renderer(RenderSettings settings) : settings_(settings) {}

RenderOutput Renderer::render(const SceneData& scene, const EmbreeScene& embreeScene) const {
    RenderOutput output;
    output.image = Image(settings_.width, settings_.height);
    output.summary.width = settings_.width;
    output.summary.height = settings_.height;
    output.summary.triangleCount = embreeScene.triangleCount();
    output.summary.lightCount = scene.lights.size();

    double luminanceSum = 0.0;

    for (int y = 0; y < settings_.height; ++y) {
        for (int x = 0; x < settings_.width; ++x) {
            const Ray ray = scene.camera.makeRay(static_cast<double>(x), static_cast<double>(y));
            ++output.summary.primaryRayCount;

            const EmbreeScene::HitRecord hit = embreeScene.intersect(ray);
            Vec3 color = scene.backgroundColor;

            if (hit.hit) {
                ++output.summary.primaryHitCount;
                color = shade(scene, embreeScene, ray, hit, output.summary);
            }

            color = clampVec3(color, 0.0, 1.0);
            output.image.setPixel(x, y, color);

            const double currentLuminance = luminance(color);
            luminanceSum += currentLuminance;
            output.summary.maxLuminance = std::max(output.summary.maxLuminance, currentLuminance);
        }
    }

    output.summary.averageLuminance = luminanceSum /
        static_cast<double>(settings_.width * settings_.height);

    for (const ControlPixel& pixel : makeControlPixels(settings_.width, settings_.height)) {
        const Ray ray = scene.camera.makeRay(static_cast<double>(pixel.x), static_cast<double>(pixel.y));
        const EmbreeScene::HitRecord hit = embreeScene.intersect(ray);

        ControlRayRecord record;
        record.name = pixel.name;
        record.pixelX = pixel.x;
        record.pixelY = pixel.y;
        record.hit = hit.hit;
        record.distance = hit.distance;
        record.geomID = hit.geomID;
        record.primID = hit.primID;

        if (hit.hit) {
            record.hitPoint = hit.position;
            record.normal = hit.normal;
            RenderSummary ignoredSummary;
            record.color = clampVec3(
                shade(scene, embreeScene, ray, hit, ignoredSummary),
                0.0,
                1.0
            );
            record.note = "Есть пересечение и выполнено локальное освещение.";
        } else {
            record.color = scene.backgroundColor;
            record.note = "Пересечения нет, использован цвет фона.";
        }

        output.controlRays.push_back(record);
    }

    return output;
}

Vec3 Renderer::shade(
    const SceneData& scene,
    const EmbreeScene& embreeScene,
    const Ray& ray,
    const EmbreeScene::HitRecord& hit,
    RenderSummary& summary
) const {
    if (hit.material == nullptr) {
        return scene.backgroundColor;
    }

    const Material& material = *hit.material;
    const Vec3 viewDirection = normalize(-ray.direction);
    const Vec3 normal = normalize(hit.normal);
    Vec3 result(0.0, 0.0, 0.0);

    for (const PointLight& light : scene.lights) {
        Vec3 toLight = light.position - hit.position;
        const double distanceSquared = std::max(toLight.lengthSquared(), 1e-8);
        const double distance = std::sqrt(distanceSquared);
        toLight /= distance;

        if (distance <= settings_.shadowBias * 2.0) {
            continue;
        }

        const double ndotl = std::max(0.0, dot(normal, toLight));
        if (ndotl <= 0.0) {
            continue;
        }

        ++summary.shadowRayCount;
        const Ray shadowRay(hit.position + normal * settings_.shadowBias, toLight);
        if (embreeScene.isOccluded(shadowRay, settings_.shadowBias, distance - settings_.shadowBias)) {
            ++summary.blockedShadowRayCount;
            continue;
        }

        const Vec3 radiance = light.intensity / distanceSquared;
        const Vec3 diffuse = hadamard(radiance, material.diffuseColor) * (material.kd * ndotl);

        double specularFactor = 0.0;
        const Vec3 halfVectorCandidate = toLight + viewDirection;
        if (halfVectorCandidate.lengthSquared() > 1e-12) {
            const Vec3 halfVector = normalize(halfVectorCandidate);
            const double ndoth = std::max(0.0, dot(normal, halfVector));
            specularFactor = material.ks * std::pow(ndoth, material.shininess);
        }
        const Vec3 specular = hadamard(radiance, material.specularColor) * specularFactor;

        result += diffuse + specular;
    }

    return result;
}
