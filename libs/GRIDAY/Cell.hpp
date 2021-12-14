#pragma once

#include <iostream>
#include <iomanip>

#include "GridayTypes.hpp"
#include "Vector.hpp"

struct Cell
    {
    Vector a; // Column 1
    Vector b; // Column 2
    Vector c; // Column 3
    };

//inline Cell inverse(const Cell& cell);
//inline Cell transpose(const Cell& cell);

//inline GReal det(const Cell& cell);

//inline Vector operator * (const Cell& cell, const Vector& v);
//inline Cell   operator * (const Cell& cell, const GReal& r);
//inline Cell   operator * (const GReal& r, const Cell&  cell);
//inline Cell   operator * (const Cell& m, const Cell& n);

//std::ostream& operator << (std::ostream& os, const Cell& cell);
//std::istream& operator >> (std::istream& is, Cell& cell);

inline
Vector
operator * (const Cell& cell, const Vector& v)
    {
    Vector c;

    c = v[0] * cell.a + v[1] * cell.b + v[2] * cell.c;

    return c;
    }

inline
Cell
operator * (const Cell& cell, const GReal& r)
    {
    Cell c;

    c.a = cell.a * r;
    c.b = cell.b * r;
    c.c = cell.c * r;

    return c;
    }

inline
Cell
operator * (const GReal& r, const Cell& cell)
    {
    return cell * r;
    }

inline
Cell
operator * (const Cell& m, const Cell& n)
    {
    return {m * n.a, m * n.b, m * n.c};
    }

inline
Cell
transpose(const Cell& cell)
    {
    return {{cell.a[0], cell.b[0], cell.c[0]},
            {cell.a[1], cell.b[1], cell.c[1]},
            {cell.a[2], cell.b[2], cell.c[2]}};
    }

inline
GReal
det(const Cell& cell)
    {
    return dot(cell.a, cross(cell.b, cell.c));
    }

inline
Cell
inverse(const Cell& cell)
    {
    Cell inv;

    GReal invd = 1.0 / det(cell);

    const Vector& a = cell.a;
    const Vector& b = cell.b;
    const Vector& c = cell.c;

    inv.a = cross(b, c) * invd;
    inv.b = cross(c, a) * invd;
    inv.c = cross(a, b) * invd;

    return transpose(inv);
    }

inline
std::ostream&
operator << (std::ostream& os, const Cell& cell)
    {
    os << std::setw(15) << cell.a[0] <<
          std::setw(15) << cell.b[0] <<
          std::setw(15) << cell.c[0] << std::endl;
    os << std::setw(15) << cell.a[1] <<
          std::setw(15) << cell.b[1] <<
          std::setw(15) << cell.c[1] << std::endl;
    os << std::setw(15) << cell.a[2] <<
          std::setw(15) << cell.b[2] <<
          std::setw(15) << cell.c[2];

    return os;
    }

inline
std::istream&
operator >> (std::istream& is, Cell& cell)
    {
    is >> cell.a >> cell.b >> cell.c;

    return is;
    }
