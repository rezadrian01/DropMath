#include <DropMath.h>

#include <chrono>
#include <iostream>

using namespace DropMath;

// Testing constructors.
void TestQuaternion_Constructors()
{
    // Default constructor (identity).
    Quaternion identity;
    assert(identity.x == 0.0f);
    assert(identity.y == 0.0f);
    assert(identity.z == 0.0f);
    assert(identity.w == 1.0f);

    // Component constructor.
    Quaternion q(1.0f, 2.0f, 3.0f, 4.0f);
    assert(q.x == 1.0f);
    assert(q.y == 2.0f);
    assert(q.z == 3.0f);
    assert(q.w == 4.0f);

    // From Vec4.
    Vec4 v(0.5f, 0.5f, 0.5f, 0.5f);
    Quaternion q2(v);
    assert(q2.x == 0.5f);
    assert(q2.y == 0.5f);
    assert(q2.z == 0.5f);
    assert(q2.w == 0.5f);
}

// Testing axis-angle constructor.
void TestQuaternion_AxisAngle()
{
    // Rotation around Y axis by 90 degrees.
    Vec3 axis = Vec3::Up();
    float angle = ToRadians(90.0f);
    Quaternion q(axis, angle);

    // Check if normalized.
    float len = q.Length();
    assert(IsZero(len - 1.0f));

    // Extract back axis and angle.
    Vec3 extractedAxis;
    float extractedAngle;
    q.ToAxisAngle(extractedAxis, extractedAngle);

    assert(IsZero(extractedAxis.x - axis.x));
    assert(IsZero(extractedAxis.y - axis.y));
    assert(IsZero(extractedAxis.z - axis.z));
    assert(IsZero(extractedAngle - angle));
}

// Testing operators.
void TestQuaternion_Operators()
{
    Quaternion q1(1.0f, 2.0f, 3.0f, 4.0f);
    Quaternion q2(5.0f, 6.0f, 7.0f, 8.0f);

    // Addition.
    Quaternion add = q1 + q2;
    assert(add.x == 6.0f);
    assert(add.y == 8.0f);
    assert(add.z == 10.0f);
    assert(add.w == 12.0f);

    // Subtraction.
    Quaternion sub = q2 - q1;
    assert(sub.x == 4.0f);
    assert(sub.y == 4.0f);
    assert(sub.z == 4.0f);
    assert(sub.w == 4.0f);

    // Scalar multiplication.
    Quaternion mul = q1 * 2.0f;
    assert(mul.x == 2.0f);
    assert(mul.y == 4.0f);
    assert(mul.z == 6.0f);
    assert(mul.w == 8.0f);

    // Scalar division.
    Quaternion div = q2 / 2.0f;
    assert(div.x == 2.5f);
    assert(div.y == 3.0f);
    assert(div.z == 3.5f);
    assert(div.w == 4.0f);

    // Equality.
    Quaternion q3(1.0f, 2.0f, 3.0f, 4.0f);
    assert(q1 == q3);
    assert(q1 != q2);
}

// Testing quaternion multiplication.
void TestQuaternion_Multiplication()
{
    // Identity * q = q.
    Quaternion identity = Quaternion::Identity();
    Quaternion q(1.0f, 2.0f, 3.0f, 4.0f);
    q.Normalize();

    Quaternion result = identity * q;
    assert(IsZero(result.x - q.x));
    assert(IsZero(result.y - q.y));
    assert(IsZero(result.z - q.z));
    assert(IsZero(result.w - q.w));

    // q * q^-1 = identity (approximately).
    Quaternion inv = q.Inverse();
    Quaternion shouldBeIdentity = q * inv;

    assert(IsZero(shouldBeIdentity.x));
    assert(IsZero(shouldBeIdentity.y));
    assert(IsZero(shouldBeIdentity.z));
    assert(IsZero(shouldBeIdentity.w - 1.0f));
}

// Testing length and normalization.
void TestQuaternion_Length()
{
    Quaternion q(1.0f, 2.0f, 3.0f, 4.0f);

    float lenSq = q.LengthSquared();
    assert(IsZero(lenSq - 30.0f)); // 1 + 4 + 9 + 16 = 30

    float len = q.Length();
    assert(IsZero(len - Sqrt(30.0f)));

    q.Normalize();
    assert(IsZero(q.Length() - 1.0f));
}

// Testing conjugate and inverse.
void TestQuaternion_ConjugateInverse()
{
    Quaternion q(1.0f, 2.0f, 3.0f, 4.0f);

    Quaternion conj = q.Conjugate();
    assert(conj.x == -1.0f);
    assert(conj.y == -2.0f);
    assert(conj.z == -3.0f);
    assert(conj.w == 4.0f);

    // For unit quaternion, inverse = conjugate.
    q.Normalize();
    Quaternion inv = q.Inverse();
    Quaternion conj2 = q.Conjugate();

    assert(IsZero(inv.x - conj2.x));
    assert(IsZero(inv.y - conj2.y));
    assert(IsZero(inv.z - conj2.z));
    assert(IsZero(inv.w - conj2.w));
}

// Testing rotation of vectors.
void TestQuaternion_RotateVector()
{
    // Rotate (1, 0, 0) by 90 degrees around Y axis should give (0, 0, -1).
    Quaternion q = Quaternion::RotationY(ToRadians(90.0f));
    Vec3 v(1.0f, 0.0f, 0.0f);
    Vec3 rotated = q.RotateVector(v);

    assert(IsZero(rotated.x));
    assert(IsZero(rotated.y));
    assert(IsZero(rotated.z + 1.0f)); // Should be -1
}

// Testing Euler angle conversion.
void TestQuaternion_Euler()
{
    // Test that FromEuler creates a valid normalized quaternion
    float pitch = ToRadians(30.0f);
    float yaw = ToRadians(45.0f);
    float roll = ToRadians(60.0f);
    
    Quaternion q = Quaternion::FromEuler(pitch, yaw, roll);
    
    // Should be normalized
    assert(IsZero(q.Length() - 1.0f));
    
    // Test that rotation matrix conversion is consistent
    Mat3x3 m = q.ToMatrix3x3();
    Quaternion q2 = Quaternion::FromMatrix(m);
    
    // Quaternions should represent same rotation (might differ in sign)
    float dot = Quaternion::Dot(q, q2);
    assert(IsZero(Abs(dot) - 1.0f));
    
    // Test ToEuler produces valid angles
    Vec3 euler = q.ToEuler();
    // Just check they're in valid range
    assert(euler.x >= -F::PI && euler.x <= F::PI);
    assert(euler.y >= -F::HALF_PI && euler.y <= F::HALF_PI);
    assert(euler.z >= -F::PI && euler.z <= F::PI);
}

// Testing matrix conversion.
void TestQuaternion_MatrixConversion()
{
    // Create quaternion from axis-angle.
    Quaternion q = Quaternion::RotationY(ToRadians(90.0f));

    // Convert to matrix.
    Mat3x3 m3 = q.ToMatrix3x3();
    Mat4x4 m4 = q.ToMatrix4x4();

    // Convert back to quaternion.
    Quaternion q2 = Quaternion::FromMatrix(m3);
    Quaternion q3 = Quaternion::FromMatrix(m4);

    // Should be approximately equal.
    assert(IsZero(Abs(q.x) - Abs(q2.x)));
    assert(IsZero(Abs(q.y) - Abs(q2.y)));
    assert(IsZero(Abs(q.z) - Abs(q2.z)));
    assert(IsZero(Abs(q.w) - Abs(q2.w)));
}

// Testing SLERP.
void TestQuaternion_Slerp()
{
    Quaternion q1 = Quaternion::RotationY(ToRadians(0.0f));
    Quaternion q2 = Quaternion::RotationY(ToRadians(90.0f));

    // At t=0, should be q1.
    Quaternion s0 = Quaternion::Slerp(q1, q2, 0.0f);
    assert(IsZero(s0.x - q1.x));
    assert(IsZero(s0.y - q1.y));
    assert(IsZero(s0.z - q1.z));
    assert(IsZero(s0.w - q1.w));

    // At t=1, should be q2.
    Quaternion s1 = Quaternion::Slerp(q1, q2, 1.0f);
    assert(IsZero(Abs(s1.x) - Abs(q2.x)));
    assert(IsZero(Abs(s1.y) - Abs(q2.y)));
    assert(IsZero(Abs(s1.z) - Abs(q2.z)));
    assert(IsZero(Abs(s1.w) - Abs(q2.w)));

    // At t=0.5, should be halfway rotation (45 degrees).
    Quaternion s05 = Quaternion::Slerp(q1, q2, 0.5f);
    Vec3 euler = s05.ToEuler();
    assert(Abs(euler.y - ToRadians(45.0f)) < 0.01f);
}

// Testing NLERP.
void TestQuaternion_Nlerp()
{
    Quaternion q1 = Quaternion::RotationY(ToRadians(0.0f));
    Quaternion q2 = Quaternion::RotationY(ToRadians(90.0f));

    Quaternion n = Quaternion::Nlerp(q1, q2, 0.5f);

    // Should be normalized.
    assert(IsZero(n.Length() - 1.0f));

    // Should be between q1 and q2.
    Vec3 euler = n.ToEuler();
    assert(euler.y > 0.0f && euler.y < ToRadians(90.0f));
}

// Testing rotation factories.
void TestQuaternion_RotationFactories()
{
    Quaternion rx = Quaternion::RotationX(ToRadians(90.0f));
    Quaternion ry = Quaternion::RotationY(ToRadians(90.0f));
    Quaternion rz = Quaternion::RotationZ(ToRadians(90.0f));

    // Each should be normalized.
    assert(IsZero(rx.Length() - 1.0f));
    assert(IsZero(ry.Length() - 1.0f));
    assert(IsZero(rz.Length() - 1.0f));

    // Test actual rotations.
    Vec3 vx = rx.RotateVector(Vec3::Up());    // Rotate Y around X.
    Vec3 vy = ry.RotateVector(Vec3::Right()); // Rotate X around Y.
    Vec3 vz = rz.RotateVector(Vec3::Right()); // Rotate X around Z.

    // Y rotated 90° around X should give Z.
    assert(IsZero(vx.x));
    assert(IsZero(vx.y));
    assert(IsZero(vx.z - 1.0f));
}

// Testing LookRotation.
void TestQuaternion_LookRotation()
{
    Vec3 forward = Vec3::Forward();
    Vec3 up = Vec3::Up();

    Quaternion q = Quaternion::LookRotation(forward, up);

    // Should be normalized.
    assert(IsZero(q.Length() - 1.0f));
}

// Testing FromToRotation.
void TestQuaternion_FromToRotation()
{
    Vec3 from = Vec3::Forward();
    Vec3 to = Vec3::Right();

    Quaternion q = Quaternion::FromToRotation(from, to);

    // Rotate 'from' by quaternion should give 'to'.
    Vec3 result = q.RotateVector(from);

    assert(IsZero(result.x - to.x));
    assert(IsZero(result.y - to.y));
    assert(IsZero(result.z - to.z));
}

// Testing dot product.
void TestQuaternion_Dot()
{
    Quaternion q1(1.0f, 2.0f, 3.0f, 4.0f);
    Quaternion q2(5.0f, 6.0f, 7.0f, 8.0f);

    float dot = Quaternion::Dot(q1, q2);
    float expected = 1 * 5 + 2 * 6 + 3 * 7 + 4 * 8; // 70
    assert(IsZero(dot - expected));
}

// Testing indexing.
void TestQuaternion_Indexing()
{
    Quaternion q(1.0f, 2.0f, 3.0f, 4.0f);

    assert(q[0] == 1.0f);
    assert(q[1] == 2.0f);
    assert(q[2] == 3.0f);
    assert(q[3] == 4.0f);

    q[0] = 10.0f;
    assert(q.x == 10.0f);
}

// Testing data pointer.
void TestQuaternion_DataPointer()
{
    Quaternion q(1.0f, 2.0f, 3.0f, 4.0f);

    const float *data = q.Data();
    assert(data[0] == 1.0f);
    assert(data[1] == 2.0f);
    assert(data[2] == 3.0f);
    assert(data[3] == 4.0f);
}

// Testing store.
void TestQuaternion_Store()
{
    Quaternion q(1.0f, 2.0f, 3.0f, 4.0f);

    float data[4];
    q.Store(data);

    assert(data[0] == 1.0f);
    assert(data[1] == 2.0f);
    assert(data[2] == 3.0f);
    assert(data[3] == 4.0f);
}

int main()
{
    using Clock = std::chrono::high_resolution_clock;
    auto start = Clock::now();

    TestQuaternion_Constructors();
    TestQuaternion_AxisAngle();
    TestQuaternion_Operators();
    TestQuaternion_Multiplication();
    TestQuaternion_Length();
    TestQuaternion_ConjugateInverse();
    TestQuaternion_RotateVector();
    TestQuaternion_Euler();
    TestQuaternion_MatrixConversion();
    TestQuaternion_Slerp();
    TestQuaternion_Nlerp();
    TestQuaternion_RotationFactories();
    TestQuaternion_LookRotation();
    TestQuaternion_FromToRotation();
    TestQuaternion_Dot();
    TestQuaternion_Indexing();
    TestQuaternion_DataPointer();
    TestQuaternion_Store();

    auto end = Clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    std::cout << "[Test Quaternion] Passed. Time: " << elapsed.count() << " ms\n";

    return 0;
}
