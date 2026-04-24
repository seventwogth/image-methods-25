#include <algorithm>
#include <clocale>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "renderer.hpp"

namespace {

std::string formatNumber(double value) {
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(6) << value;
    return stream.str();
}

std::string formatVec3(const Vec3& value) {
    return "(" + formatNumber(value.x) + ", " + formatNumber(value.y) + ", " + formatNumber(value.z) + ")";
}

void printTable(
    const std::string& title,
    const std::vector<std::string>& headers,
    const std::vector<std::vector<std::string>>& rows
) {
    std::cout << '\n' << title << '\n';

    std::vector<std::size_t> widths(headers.size(), 0U);
    for (std::size_t index = 0; index < headers.size(); ++index) {
        widths[index] = headers[index].size();
    }

    for (const auto& row : rows) {
        for (std::size_t index = 0; index < row.size(); ++index) {
            widths[index] = std::max(widths[index], row[index].size());
        }
    }

    auto printRow = [&](const std::vector<std::string>& row) {
        for (std::size_t index = 0; index < row.size(); ++index) {
            if (index > 0) {
                std::cout << " | ";
            }
            std::cout << std::left << std::setw(static_cast<int>(widths[index])) << row[index];
        }
        std::cout << '\n';
    };

    printRow(headers);
    for (std::size_t index = 0; index < widths.size(); ++index) {
        if (index > 0) {
            std::cout << "-+-";
        }
        std::cout << std::string(widths[index], '-');
    }
    std::cout << '\n';

    for (const auto& row : rows) {
        printRow(row);
    }
}

void saveCsv(
    const std::filesystem::path& outputPath,
    const std::vector<std::string>& headers,
    const std::vector<std::vector<std::string>>& rows
) {
    std::filesystem::create_directories(outputPath.parent_path());

    std::ofstream output(outputPath, std::ios::out | std::ios::trunc);
    if (!output) {
        throw std::runtime_error("Не удалось открыть CSV-файл для записи: " + outputPath.string());
    }

    auto writeRow = [&](const std::vector<std::string>& row) {
        for (std::size_t index = 0; index < row.size(); ++index) {
            if (index > 0) {
                output << ';';
            }
            output << row[index];
        }
        output << '\n';
    };

    writeRow(headers);
    for (const auto& row : rows) {
        writeRow(row);
    }
}

std::vector<std::vector<std::string>> buildCameraRows(const Camera& camera) {
    return {{
        formatNumber(camera.position.x),
        formatNumber(camera.position.y),
        formatNumber(camera.position.z),
        formatNumber(camera.target.x),
        formatNumber(camera.target.y),
        formatNumber(camera.target.z),
        formatNumber(camera.up.x),
        formatNumber(camera.up.y),
        formatNumber(camera.up.z),
        formatNumber(camera.fovYDegrees),
        std::to_string(camera.width),
        std::to_string(camera.height)
    }};
}

std::vector<std::vector<std::string>> buildLightRows(const std::vector<PointLight>& lights) {
    std::vector<std::vector<std::string>> rows;
    for (const PointLight& light : lights) {
        rows.push_back({
            light.name,
            formatNumber(light.position.x),
            formatNumber(light.position.y),
            formatNumber(light.position.z),
            formatNumber(light.intensity.x),
            formatNumber(light.intensity.y),
            formatNumber(light.intensity.z)
        });
    }
    return rows;
}

std::vector<std::vector<std::string>> buildTriangleRows(const std::vector<TriangleMesh>& meshes) {
    std::vector<std::vector<std::string>> rows;
    for (const TriangleMesh& mesh : meshes) {
        for (std::size_t triangleIndex = 0; triangleIndex < mesh.indices.size(); ++triangleIndex) {
            const auto& triangle = mesh.indices[triangleIndex];
            const Vec3& a = mesh.vertices[triangle[0]];
            const Vec3& b = mesh.vertices[triangle[1]];
            const Vec3& c = mesh.vertices[triangle[2]];
            const Vec3 normal = mesh.triangleNormal(triangleIndex);

            rows.push_back({
                mesh.name,
                mesh.name,
                std::to_string(triangleIndex),
                mesh.material.name,
                formatNumber(a.x),
                formatNumber(a.y),
                formatNumber(a.z),
                formatNumber(b.x),
                formatNumber(b.y),
                formatNumber(b.z),
                formatNumber(c.x),
                formatNumber(c.y),
                formatNumber(c.z),
                formatVec3(normal),
                formatVec3(mesh.material.diffuseColor),
                formatVec3(mesh.material.specularColor),
                formatNumber(mesh.material.kd),
                formatNumber(mesh.material.ks),
                formatNumber(mesh.material.shininess)
            });
        }
    }
    return rows;
}

std::vector<std::vector<std::string>> buildSummaryRows(const RenderSummary& summary) {
    const double hitRatio = summary.primaryRayCount == 0
        ? 0.0
        : static_cast<double>(summary.primaryHitCount) / static_cast<double>(summary.primaryRayCount);

    return {{
        std::to_string(summary.width),
        std::to_string(summary.height),
        std::to_string(summary.triangleCount),
        std::to_string(summary.lightCount),
        std::to_string(summary.primaryRayCount),
        std::to_string(summary.primaryHitCount),
        std::to_string(summary.shadowRayCount),
        std::to_string(summary.blockedShadowRayCount),
        formatNumber(hitRatio),
        formatNumber(summary.averageLuminance),
        formatNumber(summary.maxLuminance)
    }};
}

std::vector<std::vector<std::string>> buildControlRayRows(const std::vector<ControlRayRecord>& records) {
    std::vector<std::vector<std::string>> rows;
    for (const ControlRayRecord& record : records) {
        rows.push_back({
            record.name,
            std::to_string(record.pixelX),
            std::to_string(record.pixelY),
            record.hit ? "да" : "нет",
            formatNumber(record.distance),
            record.hit ? std::to_string(record.geomID) : "-",
            record.hit ? std::to_string(record.primID) : "-",
            formatVec3(record.hitPoint),
            formatVec3(record.normal),
            formatVec3(record.color),
            record.note
        });
    }
    return rows;
}

} // namespace

int main() {
    std::setlocale(LC_ALL, "");

    try {
        const std::filesystem::path outputDir = std::filesystem::path(FOURTH_OUTPUT_DIR);
        const std::filesystem::path repoDir = outputDir.parent_path().parent_path();
        const std::filesystem::path objScenePath = repoDir / "obj" / "scene.obj";
        bool useObjScene = true;

        SceneData scene;
        if (useObjScene) {
            scene = createSceneFromObj(objScenePath);
        } else {
            scene = createDefaultScene();
        }

        RenderSettings settings;
        settings.width = scene.camera.width;
        settings.height = scene.camera.height;
        settings.gamma = 2.2;
        settings.shadowBias = 1e-3;

        EmbreeScene embreeScene(scene);
        Renderer renderer(settings);
        RenderOutput renderOutput = renderer.render(scene, embreeScene);

        const std::filesystem::path renderPath = outputDir / "render.ppm";
        const std::filesystem::path cameraCsvPath = outputDir / "input_camera.csv";
        const std::filesystem::path lightsCsvPath = outputDir / "input_lights.csv";
        const std::filesystem::path trianglesCsvPath = outputDir / "input_triangles.csv";
        const std::filesystem::path summaryCsvPath = outputDir / "render_summary.csv";
        const std::filesystem::path controlRaysCsvPath = outputDir / "control_rays.csv";

        renderOutput.image.savePPM(renderPath, settings.gamma);

        const std::vector<std::string> cameraHeaders = {
            "position_x", "position_y", "position_z",
            "target_x", "target_y", "target_z",
            "up_x", "up_y", "up_z",
            "fov_y_deg", "width", "height"
        };
        const std::vector<std::string> lightHeaders = {
            "light_id", "position_x", "position_y", "position_z",
            "intensity_r", "intensity_g", "intensity_b"
        };
        const std::vector<std::string> triangleHeaders = {
            "mesh_name", "object_name", "triangle_index", "material_name",
            "v0_x", "v0_y", "v0_z",
            "v1_x", "v1_y", "v1_z",
            "v2_x", "v2_y", "v2_z",
            "normal", "diffuse_color", "specular_color",
            "kd", "ks", "shininess"
        };
        const std::vector<std::string> summaryHeaders = {
            "width", "height", "triangle_count", "light_count",
            "primary_ray_count", "primary_hit_count",
            "shadow_ray_count", "blocked_shadow_ray_count",
            "hit_ratio", "average_luminance", "max_luminance"
        };
        const std::vector<std::string> controlRayHeaders = {
            "ray_name", "pixel_x", "pixel_y", "hit",
            "distance", "geom_id", "prim_id",
            "hit_point", "normal", "color", "note"
        };

        const auto cameraRows = buildCameraRows(scene.camera);
        const auto lightRows = buildLightRows(scene.lights);
        const auto triangleRows = buildTriangleRows(scene.meshes);
        const auto summaryRows = buildSummaryRows(renderOutput.summary);
        const auto controlRayRows = buildControlRayRows(renderOutput.controlRays);

        saveCsv(cameraCsvPath, cameraHeaders, cameraRows);
        saveCsv(lightsCsvPath, lightHeaders, lightRows);
        saveCsv(trianglesCsvPath, triangleHeaders, triangleRows);
        saveCsv(summaryCsvPath, summaryHeaders, summaryRows);
        saveCsv(controlRaysCsvPath, controlRayHeaders, controlRayRows);

        printTable("Входные данные камеры", cameraHeaders, cameraRows);
        printTable("Входные данные источников света", lightHeaders, lightRows);
        printTable("Данные сцены", triangleHeaders, triangleRows);
        printTable("Результаты рендеринга", summaryHeaders, summaryRows);
        printTable("Контрольные лучи", controlRayHeaders, controlRayRows);

        std::cout
            << "\nСписок сохраненных файлов\n"
            << "- " << renderPath.string() << '\n'
            << "- " << cameraCsvPath.string() << '\n'
            << "- " << lightsCsvPath.string() << '\n'
            << "- " << trianglesCsvPath.string() << '\n'
            << "- " << summaryCsvPath.string() << '\n'
            << "- " << controlRaysCsvPath.string() << '\n';

        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Ошибка: " << error.what() << '\n';
        return 1;
    }
}
