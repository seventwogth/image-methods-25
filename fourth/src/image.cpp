#include "image.hpp"

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace {

int toByte(double value, double gamma) {
    const Vec3 clamped = clampVec3(Vec3(value, value, value), 0.0, 1.0);
    const double corrected = std::pow(clamped.x, 1.0 / gamma);
    const double scaled = corrected * 255.0;
    if (scaled <= 0.0) {
        return 0;
    }
    if (scaled >= 255.0) {
        return 255;
    }
    return static_cast<int>(scaled + 0.5);
}

} // namespace

Image::Image(int widthValue, int heightValue)
    : width_(widthValue), height_(heightValue), pixels_(static_cast<std::size_t>(widthValue * heightValue), Vec3()) {
    if (width_ <= 0 || height_ <= 0) {
        throw std::runtime_error("Image dimensions must be positive.");
    }
}

void Image::setPixel(int x, int y, const Vec3& color) {
    if (x < 0 || x >= width_ || y < 0 || y >= height_) {
        throw std::runtime_error("Pixel coordinates are out of bounds.");
    }
    pixels_[static_cast<std::size_t>(y * width_ + x)] = color;
}

const Vec3& Image::getPixel(int x, int y) const {
    if (x < 0 || x >= width_ || y < 0 || y >= height_) {
        throw std::runtime_error("Pixel coordinates are out of bounds.");
    }
    return pixels_[static_cast<std::size_t>(y * width_ + x)];
}

void Image::savePPM(const std::filesystem::path& outputPath, double gamma) const {
    if (gamma <= 0.0) {
        throw std::runtime_error("Gamma must be positive.");
    }

    std::filesystem::create_directories(outputPath.parent_path());

    std::ofstream output(outputPath, std::ios::out | std::ios::trunc);
    if (!output) {
        throw std::runtime_error("Failed to open PPM output file.");
    }

    output << "P3\n";
    output << width_ << ' ' << height_ << "\n255\n";

    for (int y = 0; y < height_; ++y) {
        for (int x = 0; x < width_; ++x) {
            const Vec3 color = clampVec3(getPixel(x, y), 0.0, 1.0);
            output
                << toByte(color.x, gamma) << ' '
                << toByte(color.y, gamma) << ' '
                << toByte(color.z, gamma) << '\n';
        }
    }
}
