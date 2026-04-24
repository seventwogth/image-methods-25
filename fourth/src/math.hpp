#pragma once

#include <algorithm>
#include <cmath>
#include <stdexcept>

struct Vec3 {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;

    Vec3() = default;
    Vec3(double xValue, double yValue, double zValue) : x(xValue), y(yValue), z(zValue) {}

    Vec3 operator+(const Vec3& other) const {
        return Vec3(x + other.x, y + other.y, z + other.z);
    }

    Vec3 operator-(const Vec3& other) const {
        return Vec3(x - other.x, y - other.y, z - other.z);
    }

    Vec3 operator-() const {
        return Vec3(-x, -y, -z);
    }

    Vec3 operator*(double scalar) const {
        return Vec3(x * scalar, y * scalar, z * scalar);
    }

    Vec3 operator/(double scalar) const {
        if (std::abs(scalar) < 1e-12) {
            throw std::runtime_error("Division by zero in Vec3.");
        }
        return Vec3(x / scalar, y / scalar, z / scalar);
    }

    Vec3& operator+=(const Vec3& other) {
        x += other.x;
        y += other.y;
        z += other.z;
        return *this;
    }

    Vec3& operator-=(const Vec3& other) {
        x -= other.x;
        y -= other.y;
        z -= other.z;
        return *this;
    }

    Vec3& operator*=(double scalar) {
        x *= scalar;
        y *= scalar;
        z *= scalar;
        return *this;
    }

    Vec3& operator/=(double scalar) {
        if (std::abs(scalar) < 1e-12) {
            throw std::runtime_error("Division by zero in Vec3.");
        }
        x /= scalar;
        y /= scalar;
        z /= scalar;
        return *this;
    }

    double lengthSquared() const {
        return x * x + y * y + z * z;
    }

    double length() const {
        return std::sqrt(lengthSquared());
    }
};

inline Vec3 operator*(double scalar, const Vec3& value) {
    return value * scalar;
}

inline double dot(const Vec3& a, const Vec3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline Vec3 cross(const Vec3& a, const Vec3& b) {
    return Vec3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

inline Vec3 hadamard(const Vec3& a, const Vec3& b) {
    return Vec3(a.x * b.x, a.y * b.y, a.z * b.z);
}

inline Vec3 normalize(const Vec3& value) {
    const double len = value.length();
    if (len < 1e-12) {
        throw std::runtime_error("Cannot normalize zero-length Vec3.");
    }
    return value / len;
}

inline Vec3 clampVec3(const Vec3& value, double minValue, double maxValue) {
    return Vec3(
        std::clamp(value.x, minValue, maxValue),
        std::clamp(value.y, minValue, maxValue),
        std::clamp(value.z, minValue, maxValue)
    );
}

inline Vec3 reflect(const Vec3& incident, const Vec3& normal) {
    return incident - normal * (2.0 * dot(incident, normal));
}

inline double luminance(const Vec3& value) {
    return 0.2126 * value.x + 0.7152 * value.y + 0.0722 * value.z;
}

struct Ray {
    Vec3 origin;
    Vec3 direction;

    Ray() = default;
    Ray(const Vec3& originValue, const Vec3& directionValue)
        : origin(originValue), direction(normalize(directionValue)) {}

    Vec3 at(double t) const {
        return origin + direction * t;
    }
};
