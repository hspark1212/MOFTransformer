#pragma once

#include <iostream>
#include <vector>
#include <array>
#include <cmath>
#include <iomanip>

#include "GridayTypes.hpp"

using Vector = std::array<GReal, 3>; // Column Vector.
using Vectors = std::vector<Vector>;

///inline GReal dot(const Vector& a, const Vector& b);
//inline GReal norm(const Vector& v);

//inline Vector cross(const Vector& a, const Vector& b);

//inline Vector operator + (const Vector& a, const Vector& b);
//inline Vector operator - (const Vector& a, const Vector& b);

//inline Vector operator * (const GReal& a, const Vector& v);
//inline Vector operator * (const Vector& v, const GReal& a);

//inline Vector operator / (const Vector& v, const GReal& a);

//std::ostream& operator << (std::ostream& os, const Vector& v);
//std::istream& operator >> (std::istream& is, Vector& v);

inline
GReal
dot(const Vector& a, const Vector& b)
    {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    }

inline
GReal
norm(const Vector& v)
    {
    return std::sqrt(dot(v, v));
    }

inline
Vector
cross(const Vector& a, const Vector& b)
    {
    return
        {
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]
        };
    }

inline
Vector
operator + (const Vector& a, const Vector& b)
    {
    Vector c;

    for (int i = 0; i < 3; ++i)
        c[i] = a[i] + b[i];

    return c;
    }

inline
Vector
operator - (const Vector& a, const Vector& b)
    {
    Vector c;

    for (int i = 0; i < 3; ++i)
        c[i] = a[i] - b[i];

    return c;
    }

inline
Vector
operator * (const GReal& a, const Vector& v)
    {
    Vector c;

    for (int i = 0; i < 3; ++i)
        c[i] = a * v[i];

    return c;
    }

inline
Vector
operator * (const Vector& v, const GReal& a)
    {
    return a * v;
    }

inline
Vector
operator / (const Vector& v, const GReal& a)
    {
    Vector c;

    for (int i = 0; i < 3; ++i)
        c[i] = v[i] / a;

    return c;
    }

inline
std::ostream&
operator << (std::ostream& os, const Vector& v)
    {
    os << std::setw(15) << v[0] <<
          std::setw(15) << v[1] <<
          std::setw(15) << v[2];

    return os;
    }

inline
std::istream&
operator >> (std::istream& is, Vector& v)
    {
    is >> v[0] >> v[1] >> v[2];

    return is;
    }
