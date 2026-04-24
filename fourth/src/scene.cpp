#include "scene.hpp"

#include <stdexcept>
#include <utility>

namespace {

struct EmbreeVertex {
    float x;
    float y;
    float z;
    float pad;
};

void checkEmbreeDevice(RTCDevice device, const std::string& context) {
    if (device == nullptr) {
        throw std::runtime_error("Embree device was not created: " + context);
    }

    const RTCError error = rtcGetDeviceError(device);
    if (error != RTC_ERROR_NONE) {
        throw std::runtime_error("Embree reported an error while " + context + ".");
    }
}

TriangleMesh makeQuadMesh(
    const std::string& name,
    const Material& material,
    const Vec3& v0,
    const Vec3& v1,
    const Vec3& v2,
    const Vec3& v3,
    const std::array<std::uint32_t, 3>& t0,
    const std::array<std::uint32_t, 3>& t1
) {
    TriangleMesh mesh;
    mesh.name = name;
    mesh.material = material;
    mesh.vertices = {v0, v1, v2, v3};
    mesh.indices = {t0, t1};
    return mesh;
}

} // namespace

std::size_t TriangleMesh::triangleCount() const {
    return indices.size();
}

Vec3 TriangleMesh::triangleNormal(std::size_t triangleIndex) const {
    const auto& triangle = indices.at(triangleIndex);
    const Vec3& a = vertices.at(triangle[0]);
    const Vec3& b = vertices.at(triangle[1]);
    const Vec3& c = vertices.at(triangle[2]);
    return normalize(cross(b - a, c - a));
}

Ray Camera::makeRay(double pixelX, double pixelY) const {
    const double aspect = static_cast<double>(width) / static_cast<double>(height);
    const double theta = fovYDegrees * 3.14159265358979323846 / 180.0;
    const double halfHeight = std::tan(theta * 0.5);
    const double halfWidth = aspect * halfHeight;

    const Vec3 forward = normalize(target - position);
    const Vec3 right = normalize(cross(forward, up));
    const Vec3 cameraUp = normalize(cross(right, forward));

    const double ndcX = (pixelX + 0.5) / static_cast<double>(width);
    const double ndcY = (pixelY + 0.5) / static_cast<double>(height);

    const double screenX = (2.0 * ndcX - 1.0) * halfWidth;
    const double screenY = (1.0 - 2.0 * ndcY) * halfHeight;

    const Vec3 direction = normalize(forward + right * screenX + cameraUp * screenY);
    return Ray(position, direction);
}

EmbreeScene::EmbreeScene(const SceneData& data) : data_(&data) {
    device_ = rtcNewDevice(nullptr);
    checkEmbreeDevice(device_, "creating the device");

    scene_ = rtcNewScene(device_);
    if (scene_ == nullptr) {
        throw std::runtime_error("Embree scene was not created.");
    }

    for (const TriangleMesh& mesh : data.meshes) {
        RTCGeometry geometry = rtcNewGeometry(device_, RTC_GEOMETRY_TYPE_TRIANGLE);
        if (geometry == nullptr) {
            throw std::runtime_error("Failed to create Embree triangle geometry.");
        }

        auto* vertices = static_cast<EmbreeVertex*>(
            rtcSetNewGeometryBuffer(
                geometry,
                RTC_BUFFER_TYPE_VERTEX,
                0,
                RTC_FORMAT_FLOAT3,
                sizeof(EmbreeVertex),
                mesh.vertices.size()
            )
        );

        for (std::size_t index = 0; index < mesh.vertices.size(); ++index) {
            vertices[index].x = static_cast<float>(mesh.vertices[index].x);
            vertices[index].y = static_cast<float>(mesh.vertices[index].y);
            vertices[index].z = static_cast<float>(mesh.vertices[index].z);
            vertices[index].pad = 0.0f;
        }

        auto* triangles = static_cast<std::uint32_t*>(
            rtcSetNewGeometryBuffer(
                geometry,
                RTC_BUFFER_TYPE_INDEX,
                0,
                RTC_FORMAT_UINT3,
                3 * sizeof(std::uint32_t),
                mesh.indices.size()
            )
        );

        for (std::size_t index = 0; index < mesh.indices.size(); ++index) {
            triangles[index * 3 + 0] = mesh.indices[index][0];
            triangles[index * 3 + 1] = mesh.indices[index][1];
            triangles[index * 3 + 2] = mesh.indices[index][2];
        }

        rtcCommitGeometry(geometry);
        const unsigned int geomID = rtcAttachGeometry(scene_, geometry);
        rtcReleaseGeometry(geometry);

        if (geomID >= geometryBindings_.size()) {
            geometryBindings_.resize(static_cast<std::size_t>(geomID) + 1U);
        }
        geometryBindings_[geomID].mesh = &mesh;
    }

    rtcCommitScene(scene_);
    checkEmbreeDevice(device_, "committing the scene");
}

EmbreeScene::~EmbreeScene() {
    release();
}

EmbreeScene::EmbreeScene(EmbreeScene&& other) noexcept {
    *this = std::move(other);
}

EmbreeScene& EmbreeScene::operator=(EmbreeScene&& other) noexcept {
    if (this != &other) {
        release();
        device_ = other.device_;
        scene_ = other.scene_;
        data_ = other.data_;
        geometryBindings_ = std::move(other.geometryBindings_);
        other.device_ = nullptr;
        other.scene_ = nullptr;
        other.data_ = nullptr;
    }
    return *this;
}

EmbreeScene::HitRecord EmbreeScene::intersect(const Ray& ray, double tMin, double tMax) const {
    RTCIntersectContext context;
    rtcInitIntersectContext(&context);

    RTCRayHit rayHit{};
    rayHit.ray.org_x = static_cast<float>(ray.origin.x);
    rayHit.ray.org_y = static_cast<float>(ray.origin.y);
    rayHit.ray.org_z = static_cast<float>(ray.origin.z);
    rayHit.ray.dir_x = static_cast<float>(ray.direction.x);
    rayHit.ray.dir_y = static_cast<float>(ray.direction.y);
    rayHit.ray.dir_z = static_cast<float>(ray.direction.z);
    rayHit.ray.tnear = static_cast<float>(tMin);
    rayHit.ray.tfar = static_cast<float>(tMax);
    rayHit.ray.mask = 0xFFFFFFFFu;
    rayHit.ray.flags = 0u;
    rayHit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
    rayHit.hit.primID = RTC_INVALID_GEOMETRY_ID;

    rtcIntersect1(scene_, &context, &rayHit);

    if (rayHit.hit.geomID == RTC_INVALID_GEOMETRY_ID) {
        return {};
    }

    const GeometryBinding& binding = geometryBindings_.at(rayHit.hit.geomID);
    const TriangleMesh* mesh = binding.mesh;
    if (mesh == nullptr) {
        throw std::runtime_error("Embree geometry binding is missing.");
    }

    Vec3 normal = mesh->triangleNormal(rayHit.hit.primID);
    if (dot(normal, ray.direction) > 0.0) {
        normal = -normal;
    }

    HitRecord record;
    record.hit = true;
    record.distance = static_cast<double>(rayHit.ray.tfar);
    record.position = ray.at(record.distance);
    record.normal = normal;
    record.material = &mesh->material;
    record.geomID = rayHit.hit.geomID;
    record.primID = rayHit.hit.primID;
    return record;
}

bool EmbreeScene::isOccluded(const Ray& ray, double tMin, double tMax) const {
    RTCIntersectContext context;
    rtcInitIntersectContext(&context);

    RTCRay shadowRay{};
    shadowRay.org_x = static_cast<float>(ray.origin.x);
    shadowRay.org_y = static_cast<float>(ray.origin.y);
    shadowRay.org_z = static_cast<float>(ray.origin.z);
    shadowRay.dir_x = static_cast<float>(ray.direction.x);
    shadowRay.dir_y = static_cast<float>(ray.direction.y);
    shadowRay.dir_z = static_cast<float>(ray.direction.z);
    shadowRay.tnear = static_cast<float>(tMin);
    shadowRay.tfar = static_cast<float>(tMax);
    shadowRay.mask = 0xFFFFFFFFu;
    shadowRay.flags = 0u;

    rtcOccluded1(scene_, &context, &shadowRay);
    return shadowRay.tfar < 0.0f;
}

std::size_t EmbreeScene::triangleCount() const {
    std::size_t total = 0;
    for (const TriangleMesh& mesh : data_->meshes) {
        total += mesh.triangleCount();
    }
    return total;
}

void EmbreeScene::release() {
    if (scene_ != nullptr) {
        rtcReleaseScene(scene_);
        scene_ = nullptr;
    }
    if (device_ != nullptr) {
        rtcReleaseDevice(device_);
        device_ = nullptr;
    }
}

SceneData createDefaultScene() {
    SceneData scene;
    scene.camera.position = Vec3(0.0, 1.6, 4.8);
    scene.camera.target = Vec3(0.0, 1.0, 0.2);
    scene.camera.up = Vec3(0.0, 1.0, 0.0);
    scene.camera.fovYDegrees = 45.0;
    scene.camera.width = 640;
    scene.camera.height = 640;
    scene.backgroundColor = Vec3(0.02, 0.03, 0.05);

    scene.lights = {
        PointLight{"L1_warm", Vec3(-1.8, 3.5, 2.0), Vec3(42.0, 32.0, 24.0)},
        PointLight{"L2_cool", Vec3(2.2, 2.6, -0.5), Vec3(20.0, 24.0, 36.0)}
    };

    const Material floorMaterial{
        "floor",
        Vec3(0.72, 0.72, 0.74),
        Vec3(0.18, 0.18, 0.18),
        0.90,
        0.10,
        12.0
    };
    const Material backWallMaterial{
        "back_wall",
        Vec3(0.55, 0.62, 0.82),
        Vec3(0.10, 0.10, 0.16),
        0.92,
        0.08,
        10.0
    };
    const Material sideWallMaterial{
        "side_wall",
        Vec3(0.82, 0.58, 0.52),
        Vec3(0.15, 0.10, 0.10),
        0.90,
        0.10,
        10.0
    };
    const Material pyramidMaterial{
        "pyramid",
        Vec3(0.88, 0.72, 0.32),
        Vec3(0.95, 0.92, 0.70),
        0.70,
        0.30,
        36.0
    };

    scene.meshes.push_back(
        makeQuadMesh(
            "floor",
            floorMaterial,
            Vec3(-2.5, 0.0, -2.5),
            Vec3(2.5, 0.0, -2.5),
            Vec3(2.5, 0.0, 2.5),
            Vec3(-2.5, 0.0, 2.5),
            {0, 2, 1},
            {0, 3, 2}
        )
    );

    scene.meshes.push_back(
        makeQuadMesh(
            "back_wall",
            backWallMaterial,
            Vec3(-2.5, 0.0, -2.5),
            Vec3(-2.5, 3.0, -2.5),
            Vec3(2.5, 3.0, -2.5),
            Vec3(2.5, 0.0, -2.5),
            {0, 2, 1},
            {0, 3, 2}
        )
    );

    scene.meshes.push_back(
        makeQuadMesh(
            "left_wall",
            sideWallMaterial,
            Vec3(-2.5, 0.0, 2.5),
            Vec3(-2.5, 3.0, 2.5),
            Vec3(-2.5, 3.0, -2.5),
            Vec3(-2.5, 0.0, -2.5),
            {0, 2, 1},
            {0, 3, 2}
        )
    );

    TriangleMesh pyramid;
    pyramid.name = "pyramid";
    pyramid.material = pyramidMaterial;
    pyramid.vertices = {
        Vec3(-0.8, 0.0, -0.2),
        Vec3(0.8, 0.0, -0.2),
        Vec3(0.8, 0.0, 1.2),
        Vec3(-0.8, 0.0, 1.2),
        Vec3(0.0, 1.6, 0.45)
    };
    pyramid.indices = {
        {3, 2, 4},
        {1, 0, 4},
        {0, 3, 4},
        {2, 1, 4},
        {0, 1, 2},
        {0, 2, 3}
    };
    scene.meshes.push_back(pyramid);

    return scene;
}
