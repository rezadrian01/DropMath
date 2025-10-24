#include <cmath>

namespace DropMath
{
    namespace
    {
        const float4 g_QUAT_EPSILON_F = _mm_set1_ps(F::EPSILON);
        const float4 g_QUAT_SIGN_MASK_F = _mm_castsi128_ps(_mm_set1_epi32(F::SIGN_MASK));
    } // anonymous namespace

    // Constructor from axis and angle.
    inline Quaternion::Quaternion(const Vec3 &axis, float angleRadians)
    {
        float halfAngle = angleRadians * 0.5f;
        float s = Sin(halfAngle);
        float c = Cos(halfAngle);

        x = axis.x * s;
        y = axis.y * s;
        z = axis.z * s;
        w = c;
    }

    // From Euler angles (pitch around X, yaw around Y, roll around Z).
    inline Quaternion Quaternion::FromEuler(float pitch, float yaw, float roll)
    {
        float cy = Cos(yaw * 0.5f);
        float sy = Sin(yaw * 0.5f);
        float cp = Cos(pitch * 0.5f);
        float sp = Sin(pitch * 0.5f);
        float cr = Cos(roll * 0.5f);
        float sr = Sin(roll * 0.5f);

        Quaternion q;
        q.w = cr * cp * cy + sr * sp * sy;
        q.x = sr * cp * cy - cr * sp * sy;
        q.y = cr * sp * cy + sr * cp * sy;
        q.z = cr * cp * sy - sr * sp * cy;

        return q;
    }

    // From rotation matrix (3x3).
    inline Quaternion Quaternion::FromMatrix(const Mat3x3 &m)
    {
        Quaternion q;
        float trace = m[0][0] + m[1][1] + m[2][2];

        if (trace > 0.0f)
        {
            float s = Sqrt(trace + 1.0f) * 2.0f; // s = 4 * qw
            q.w = 0.25f * s;
            q.x = (m[2][1] - m[1][2]) / s;
            q.y = (m[0][2] - m[2][0]) / s;
            q.z = (m[1][0] - m[0][1]) / s;
        }
        else if ((m[0][0] > m[1][1]) && (m[0][0] > m[2][2]))
        {
            float s = Sqrt(1.0f + m[0][0] - m[1][1] - m[2][2]) * 2.0f; // s = 4 * qx
            q.w = (m[2][1] - m[1][2]) / s;
            q.x = 0.25f * s;
            q.y = (m[0][1] + m[1][0]) / s;
            q.z = (m[0][2] + m[2][0]) / s;
        }
        else if (m[1][1] > m[2][2])
        {
            float s = Sqrt(1.0f + m[1][1] - m[0][0] - m[2][2]) * 2.0f; // s = 4 * qy
            q.w = (m[0][2] - m[2][0]) / s;
            q.x = (m[0][1] + m[1][0]) / s;
            q.y = 0.25f * s;
            q.z = (m[1][2] + m[2][1]) / s;
        }
        else
        {
            float s = Sqrt(1.0f + m[2][2] - m[0][0] - m[1][1]) * 2.0f; // s = 4 * qz
            q.w = (m[1][0] - m[0][1]) / s;
            q.x = (m[0][2] + m[2][0]) / s;
            q.y = (m[1][2] + m[2][1]) / s;
            q.z = 0.25f * s;
        }

        return q;
    }

    // From rotation matrix (4x4) - extract 3x3 rotation part.
    inline Quaternion Quaternion::FromMatrix(const Mat4x4 &m)
    {
        Mat3x3 m3(
            Vec3(m[0].x, m[0].y, m[0].z),
            Vec3(m[1].x, m[1].y, m[1].z),
            Vec3(m[2].x, m[2].y, m[2].z));
        return FromMatrix(m3);
    }

    // Access operators.
    inline float &Quaternion::operator[](int i)
    {
        assert(i >= 0 && i < 4);
        return array[i];
    }

    inline const float &Quaternion::operator[](int i) const
    {
        assert(i >= 0 && i < 4);
        return array[i];
    }

    // Arithmetic operators.
    inline Quaternion Quaternion::operator+(const Quaternion &q) const
    {
        return Quaternion(_mm_add_ps(v, q.v));
    }

    inline Quaternion Quaternion::operator-(const Quaternion &q) const
    {
        return Quaternion(_mm_sub_ps(v, q.v));
    }

    // Quaternion multiplication (Hamilton product).
    inline Quaternion Quaternion::operator*(const Quaternion &q) const
    {
        Quaternion result;
        result.w = w * q.w - x * q.x - y * q.y - z * q.z;
        result.x = w * q.x + x * q.w + y * q.z - z * q.y;
        result.y = w * q.y - x * q.z + y * q.w + z * q.x;
        result.z = w * q.z + x * q.y - y * q.x + z * q.w;
        return result;
    }

    inline Quaternion Quaternion::operator*(float s) const
    {
        return Quaternion(_mm_mul_ps(v, _mm_set1_ps(s)));
    }

    inline Quaternion Quaternion::operator/(float s) const
    {
        return Quaternion(_mm_div_ps(v, _mm_set1_ps(s)));
    }

    inline bool Quaternion::operator==(const Quaternion &q) const
    {
        float4 delta = _mm_sub_ps(q.v, v);
        float4 abs = _mm_and_ps(g_QUAT_SIGN_MASK_F, delta);
        float4 cmp = _mm_cmplt_ps(abs, g_QUAT_EPSILON_F);
        int mask = _mm_movemask_ps(cmp);
        return mask == 0xF;
    }

    inline bool Quaternion::operator!=(const Quaternion &q) const
    {
        return !(*this == q);
    }

    // Length operations.
    inline float Quaternion::Length() const
    {
        float4 dot = _mm_dp_ps(v, v, 0b11110001);
        float4 sqrt = _mm_sqrt_ps(dot);
        return _mm_cvtss_f32(sqrt);
    }

    inline float Quaternion::LengthSquared() const
    {
        float4 dot = _mm_dp_ps(v, v, 0b11110001);
        return _mm_cvtss_f32(dot);
    }

    inline void Quaternion::Normalize()
    {
        float len = Length();
        if (len > F::EPSILON)
        {
            v = _mm_div_ps(v, _mm_set1_ps(len));
        }
    }

    inline Quaternion Quaternion::Normalized() const
    {
        Quaternion result = *this;
        result.Normalize();
        return result;
    }

    // Conjugate (negate vector part).
    inline Quaternion Quaternion::Conjugate() const
    {
        Quaternion result;
        result.x = -x;
        result.y = -y;
        result.z = -z;
        result.w = w;
        return result;
    }

    // Inverse.
    inline Quaternion Quaternion::Inverse() const
    {
        float lenSq = LengthSquared();
        if (lenSq < F::EPSILON)
            return Identity();

        Quaternion conj = Conjugate();
        return conj / lenSq;
    }

    // Rotate a vector by this quaternion.
    inline Vec3 Quaternion::RotateVector(const Vec3 &vec) const
    {
        // Using formula: v' = q * v * q^-1
        // Optimized version: v' = v + 2 * cross(q.xyz, cross(q.xyz, v) + q.w * v)
        Vec3 qvec(x, y, z);
        Vec3 cross1 = Vec3::Cross(qvec, vec);
        Vec3 cross2 = Vec3::Cross(qvec, cross1 + vec * w);
        return vec + cross2 * 2.0f;
    }

    // Convert to rotation matrix (3x3).
    inline Mat3x3 Quaternion::ToMatrix3x3() const
    {
        float xx = x * x;
        float yy = y * y;
        float zz = z * z;
        float xy = x * y;
        float xz = x * z;
        float yz = y * z;
        float wx = w * x;
        float wy = w * y;
        float wz = w * z;

        Mat3x3 m;
        m[0][0] = 1.0f - 2.0f * (yy + zz);
        m[0][1] = 2.0f * (xy - wz);
        m[0][2] = 2.0f * (xz + wy);

        m[1][0] = 2.0f * (xy + wz);
        m[1][1] = 1.0f - 2.0f * (xx + zz);
        m[1][2] = 2.0f * (yz - wx);

        m[2][0] = 2.0f * (xz - wy);
        m[2][1] = 2.0f * (yz + wx);
        m[2][2] = 1.0f - 2.0f * (xx + yy);

        return m;
    }

    // Convert to rotation matrix (4x4).
    inline Mat4x4 Quaternion::ToMatrix4x4() const
    {
        float xx = x * x;
        float yy = y * y;
        float zz = z * z;
        float xy = x * y;
        float xz = x * z;
        float yz = y * z;
        float wx = w * x;
        float wy = w * y;
        float wz = w * z;

        return Mat4x4(
            Vec4(1.0f - 2.0f * (yy + zz), 2.0f * (xy - wz), 2.0f * (xz + wy), 0.0f),
            Vec4(2.0f * (xy + wz), 1.0f - 2.0f * (xx + zz), 2.0f * (yz - wx), 0.0f),
            Vec4(2.0f * (xz - wy), 2.0f * (yz + wx), 1.0f - 2.0f * (xx + yy), 0.0f),
            Vec4(0.0f, 0.0f, 0.0f, 1.0f));
    }

    // Convert to Euler angles (pitch, yaw, roll).
    inline Vec3 Quaternion::ToEuler() const
    {
        Vec3 euler;

        // Roll (x-axis rotation).
        float sinr_cosp = 2.0f * (w * x + y * z);
        float cosr_cosp = 1.0f - 2.0f * (x * x + y * y);
        euler.x = atan2f(sinr_cosp, cosr_cosp);

        // Pitch (y-axis rotation).
        float sinp = 2.0f * (w * y - z * x);
        if (Abs(sinp) >= 1.0f)
            euler.y = copysignf(F::HALF_PI, sinp); // Use 90 degrees if out of range.
        else
            euler.y = asinf(sinp);

        // Yaw (z-axis rotation).
        float siny_cosp = 2.0f * (w * z + x * y);
        float cosy_cosp = 1.0f - 2.0f * (y * y + z * z);
        euler.z = atan2f(siny_cosp, cosy_cosp);

        return euler;
    }

    // Get axis and angle.
    inline void Quaternion::ToAxisAngle(Vec3 &axis, float &angle) const
    {
        Quaternion q = Normalized();
        angle = 2.0f * acosf(Clamp(q.w, -1.0f, 1.0f));

        float s = Sqrt(1.0f - q.w * q.w);
        if (s < F::EPSILON)
        {
            // Angle is 0 or 360 degrees, axis doesn't matter.
            axis = Vec3(1.0f, 0.0f, 0.0f);
        }
        else
        {
            axis = Vec3(q.x / s, q.y / s, q.z / s);
        }
    }

    // Dot product.
    inline float Quaternion::Dot(const Quaternion &a, const Quaternion &b)
    {
        float4 dot = _mm_dp_ps(a.v, b.v, 0b11110001);
        return _mm_cvtss_f32(dot);
    }

    // Linear interpolation.
    inline Quaternion Quaternion::Lerp(const Quaternion &a, const Quaternion &b, float t)
    {
        return Quaternion(DropMath::Lerp(Vec4(a.v), Vec4(b.v), t));
    }

    // Spherical linear interpolation (SLERP).
    inline Quaternion Quaternion::Slerp(const Quaternion &a, const Quaternion &b, float t)
    {
        Quaternion q1 = a.Normalized();
        Quaternion q2 = b.Normalized();

        float dot = Dot(q1, q2);

        // If dot is negative, negate q2 to take shorter path.
        if (dot < 0.0f)
        {
            q2 = q2 * -1.0f;
            dot = -dot;
        }

        // If quaternions are very close, use linear interpolation.
        if (dot > 0.9995f)
        {
            return Lerp(q1, q2, t).Normalized();
        }

        // Clamp dot to valid range for acos.
        dot = Clamp(dot, -1.0f, 1.0f);

        float theta = acosf(dot);
        float sinTheta = Sin(theta);

        float w1 = Sin((1.0f - t) * theta) / sinTheta;
        float w2 = Sin(t * theta) / sinTheta;

        return q1 * w1 + q2 * w2;
    }

    // Normalized linear interpolation (NLERP).
    inline Quaternion Quaternion::Nlerp(const Quaternion &a, const Quaternion &b, float t)
    {
        Quaternion q1 = a.Normalized();
        Quaternion q2 = b.Normalized();

        float dot = Dot(q1, q2);

        // Take shorter path.
        if (dot < 0.0f)
        {
            q2 = q2 * -1.0f;
        }

        return Lerp(q1, q2, t).Normalized();
    }

    // Rotation around X axis.
    inline Quaternion Quaternion::RotationX(float angleRadians)
    {
        float halfAngle = angleRadians * 0.5f;
        return Quaternion(Sin(halfAngle), 0.0f, 0.0f, Cos(halfAngle));
    }

    // Rotation around Y axis.
    inline Quaternion Quaternion::RotationY(float angleRadians)
    {
        float halfAngle = angleRadians * 0.5f;
        return Quaternion(0.0f, Sin(halfAngle), 0.0f, Cos(halfAngle));
    }

    // Rotation around Z axis.
    inline Quaternion Quaternion::RotationZ(float angleRadians)
    {
        float halfAngle = angleRadians * 0.5f;
        return Quaternion(0.0f, 0.0f, Sin(halfAngle), Cos(halfAngle));
    }

    // Look rotation.
    inline Quaternion Quaternion::LookRotation(const Vec3 &forward, const Vec3 &up)
    {
        Vec3 f = forward;
        f.Normalize();

        Vec3 r = Vec3::Cross(up, f);
        r.Normalize();

        Vec3 u = Vec3::Cross(f, r);

        // Build rotation matrix.
        Mat3x3 m(
            r, // Right
            u, // Up
            f  // Forward
        );

        return FromMatrix(m);
    }

    // From-to rotation.
    inline Quaternion Quaternion::FromToRotation(const Vec3 &from, const Vec3 &to)
    {
        Vec3 f = from;
        Vec3 t = to;
        f.Normalize();
        t.Normalize();

        float dot = Vec3::Dot(f, t);

        // Vectors are parallel and same direction.
        if (dot >= 0.999999f)
        {
            return Identity();
        }

        // Vectors are parallel but opposite direction.
        if (dot <= -0.999999f)
        {
            // Find an axis perpendicular to from.
            Vec3 axis = Vec3::Cross(Vec3::Right(), f);
            if (axis.LengthSquared() < F::EPSILON)
            {
                axis = Vec3::Cross(Vec3::Up(), f);
            }
            axis.Normalize();
            return Quaternion(axis, F::PI);
        }

        // General case.
        Vec3 axis = Vec3::Cross(f, t);
        axis.Normalize();

        float angle = acosf(dot);
        return Quaternion(axis, angle);
    }

} // namespace DropMath
