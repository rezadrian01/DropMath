#pragma once

#include "../vec/DM_Vec3.h"
#include "../vec/DM_Vec4.h"
#include "../mat/DM_Mat3x3.h"
#include "../mat/DM_Mat4x4.h"

namespace DropMath
{
    // Quaternion for 3D rotations.
    // Represented as (x, y, z, w) where w is the scalar part.
    // Uses SIMD acceleration via Vec4 internally.
    struct alignas(16) Quaternion
    {
        union
        {
            float4 v; // Internal SIMD representation.
            struct
            {
                float x, y, z, w;
            };
            float array[4];
        };

        // Default constructor creates identity quaternion (0, 0, 0, 1).
        Quaternion() : v(_mm_set_ps(1.0f, 0.0f, 0.0f, 0.0f)) {}

        // Construct quaternion from components.
        Quaternion(float x, float y, float z, float w) : v(_mm_set_ps(w, z, y, x)) {}

        // Construct from Vec4 (x, y, z, w).
        explicit Quaternion(const Vec4 &v) : v(v.v) {}

        // Construct from axis and angle (in radians).
        Quaternion(const Vec3 &axis, float angleRadians);

        // Construct from Euler angles (pitch, yaw, roll in radians).
        static Quaternion FromEuler(float pitch, float yaw, float roll);

        // Construct from rotation matrix.
        static Quaternion FromMatrix(const Mat3x3 &m);
        static Quaternion FromMatrix(const Mat4x4 &m);

        // Access operators.
        float &operator[](int i);
        const float &operator[](int i) const;

        // Arithmetic operators.
        Quaternion operator+(const Quaternion &q) const;
        Quaternion operator-(const Quaternion &q) const;
        Quaternion operator*(const Quaternion &q) const; // Quaternion multiplication.
        Quaternion operator*(float s) const;             // Scalar multiplication.
        Quaternion operator/(float s) const;             // Scalar division.
        bool operator==(const Quaternion &q) const;
        bool operator!=(const Quaternion &q) const;

        // Return data pointer.
        float *Data() { return &x; }
        const float *Data() const { return &x; }

        // Length (magnitude) of the quaternion.
        float Length() const;

        // Squared length (more efficient than Length).
        float LengthSquared() const;

        // Normalize the quaternion to unit length.
        void Normalize();

        // Return normalized copy of this quaternion.
        Quaternion Normalized() const;

        // Conjugate of the quaternion (negate x, y, z).
        Quaternion Conjugate() const;

        // Inverse of the quaternion (conjugate / length squared).
        Quaternion Inverse() const;

        // Store quaternion to float array.
        void Store(float *dst) const { _mm_storeu_ps(dst, v); }

        // Rotate a vector by this quaternion.
        Vec3 RotateVector(const Vec3 &vec) const;

        // Convert to rotation matrix.
        Mat3x3 ToMatrix3x3() const;
        Mat4x4 ToMatrix4x4() const;

        // Convert to Euler angles (pitch, yaw, roll in radians).
        Vec3 ToEuler() const;

        // Get rotation axis and angle.
        void ToAxisAngle(Vec3 &axis, float &angle) const;

        // Dot product of two quaternions.
        static float Dot(const Quaternion &a, const Quaternion &b);

        // Linear interpolation between two quaternions.
        static Quaternion Lerp(const Quaternion &a, const Quaternion &b, float t);

        // Spherical linear interpolation (SLERP) between two quaternions.
        // This is the preferred method for smooth rotation interpolation.
        static Quaternion Slerp(const Quaternion &a, const Quaternion &b, float t);

        // Normalized linear interpolation (NLERP) - faster approximation of SLERP.
        static Quaternion Nlerp(const Quaternion &a, const Quaternion &b, float t);

        // Identity quaternion (0, 0, 0, 1).
        static Quaternion Identity() { return Quaternion(); }

        // Quaternion representing rotation around X axis.
        static Quaternion RotationX(float angleRadians);

        // Quaternion representing rotation around Y axis.
        static Quaternion RotationY(float angleRadians);

        // Quaternion representing rotation around Z axis.
        static Quaternion RotationZ(float angleRadians);

        // Look rotation - creates quaternion that rotates forward vector to look at target.
        static Quaternion LookRotation(const Vec3 &forward, const Vec3 &up = Vec3::Up());

        // Rotate from one direction to another.
        static Quaternion FromToRotation(const Vec3 &from, const Vec3 &to);

        explicit Quaternion(const float4 &v) : v(v) {}
    };

} // namespace DropMath

#include "DM_Quaternion.inl"
