#pragma once

#include <filesystem>
#include <vector>

#include "math.hpp"

class Image {
public:
    Image() = default;
    Image(int widthValue, int heightValue);

    int width() const { return width_; }
    int height() const { return height_; }

    void setPixel(int x, int y, const Vec3& color);
    const Vec3& getPixel(int x, int y) const;

    void savePPM(const std::filesystem::path& outputPath, double gamma) const;

private:
    int width_ = 0;
    int height_ = 0;
    std::vector<Vec3> pixels_;
};
