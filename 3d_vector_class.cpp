// https://github.com/OmarAflak/RayTracer-CPP/blob/master/main.cpp

#ifndef VECT3_H
#define VECT3_H

#include <iostream>
#include <sstream>
#include <cmath>

class Vect3 {
public:
    float x, y, z;

    Vect3(const float& x, const float& y, const float& z);
    Vect3(const Vect3& v);
    Vect3();

    void set(const Vect3& v);
    void set(const float& x, const float& y, const float& z);

    float dot(const Vect3& v) const;
    Vect3 cross(const Vect3& v) const;
    Vect3 elementWiseProduct(const Vect3& v) const;
    Vect3 normalized() const;
    Vect3 reflected(const Vect3& axis) const;
    float length() const;

    Vect3 plus(const Vect3& v) const;
    Vect3 minus(const Vect3& v) const;
    Vect3 times(const float& f) const;
    std::string toString() const;
};

std::ostream& operator<<(std::ostream& os, const Vect3& v);
Vect3 operator+(const Vect3& a, const Vect3& b);
Vect3 operator-(const Vect3& a, const Vect3& b);
Vect3 operator*(const Vect3& a, const Vect3& b);
Vect3 operator*(const float& f, const Vect3& v);

#endif


#include "vect3.h"

Vect3::Vect3(const float& x, const float& y, const float& z) {
    set(x, y, z);
}

Vect3::Vect3(const Vect3& v) {
    set(v);
}

Vect3::Vect3() {
    set(0, 0, 0);
}

void Vect3::set(const Vect3& v) {
    set(v.x, v.y, v.z);
}

void Vect3::set(const float& x, const float& y, const float& z) {
    this->x = x;
    this->y = y;
    this->z = z;
}

float Vect3::dot(const Vect3& v) const {
    return x * v.x + y * v.y + z * v.z;
}

Vect3 Vect3::elementWiseProduct(const Vect3& v) const {
    return Vect3(x * v.x, y * v.y, z * v.z);
}

Vect3 Vect3::cross(const Vect3& v) const {
    return Vect3(
        y * v.z - z * v.y,
        z * v.x - x * v.z,
        x * v.y - y * v.x
    );
}

Vect3 Vect3::normalized() const {
    float norm = length();
    if (norm > 0) {
        norm = 1 / norm;
        return times(norm);
    }
    return Vect3();
}

Vect3 Vect3::reflected(const Vect3& axis) const {
    return *this - 2 * this->dot(axis) * axis;
}

float Vect3::length() const {
    return sqrt(dot(*this));
}

Vect3 Vect3::plus(const Vect3& v) const {
    return Vect3(x + v.x, y + v.y, z + v.z);
}

Vect3 Vect3::minus(const Vect3& v) const {
    return Vect3(x - v.x, y - v.y, z - v.z);
}

Vect3 Vect3::times(const float& f) const {
    return Vect3(f * x, f * y, f * z);
}

std::string Vect3::toString() const {
    std::stringstream ss;
    ss << "[" << x << ", " << y << ", " << z << "]";
    return ss.str();
}

std::ostream& operator<<(std::ostream& os, const Vect3& v) {
    os << v.toString();
    return os;
}

Vect3 operator+(const Vect3& a, const Vect3& b) {
    return a.plus(b);
}

Vect3 operator-(const Vect3& a, const Vect3& b) {
    return a.minus(b);
}

Vect3 operator*(const Vect3& a, const Vect3& b) {
    return a.elementWiseProduct(b);
}

Vect3 operator*(const float& f, const Vect3& v) {
    return v.times(f);
}
