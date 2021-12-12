#include "Gaussian.hpp"

#include <iostream>
#include <iomanip>
#include <limits>
#include <cmath>


#include "GridayException.hpp"

Gaussian::Gaussian(GReal scale, GReal width, GReal rcut) :
    PairEnergy {},
    mScale {scale},
    mWidth {width},
    //mSigSq {sig * sig},
    mRcut {rcut},
    mRcutSq {rcut * rcut}
    {

    }

void
Gaussian::setSimulationBox(const Cell& box)
    {
    mBox = box;
    mInvBox = inverse(box);
    }

GReal
Gaussian::calculate(const Vectors& r1, const Vectors& r2)
    {
    GReal zero = static_cast<GReal>(0.001);

    bool cond1 = mScale  < zero;
    bool cond2 = mWidth  < zero;
    bool cond3 = mRcut   < zero;

    if (cond1 || cond2 || cond3)
        return static_cast<GReal>(0.0);

    if (r1.size() == 0 || r2.size() == 0)
        return static_cast<GReal>(0.0);

    int size1 = r1.size();
    int size2 = r2.size();

    //GReal overlapDistSq = 1e-8;
    //GReal one = 1.0;
    GReal energy = 0.0;

    for (int i = 0; i < size1; ++i)
        {
        const Vector& ri = r1[i];

        for (int j = 0; j < size2; ++j)
            {
            const Vector& rj = r2[j];

            auto s = mInvBox * (ri - rj);

            for (auto& si : s)
                {
                if (si >  0.5)
                    si -= 1.0;
                if (si < -0.5)
                    si += 1.0;
                }

            auto r = mBox * s;
            auto rsq = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];

            if (rsq < mRcutSq)
                {
                //if (rsq < overlapDistSq)
                    //return std::numeric_limits<GReal>::max();
                energy += mScale * exp(-rsq / mWidth);
                }
            }
        }

    return energy;
    }

std::unique_ptr<PairEnergy>
Gaussian::clone()
    {
    return std::make_unique<Gaussian>(mScale, mWidth, mRcut);
    }

void
Gaussian::print()
    {
    using namespace std;

    cout << "Pair Type: Gaussian, Parameter: " <<
                 setw(10) << "Scale = "  << setw(10) << mScale <<
                 setw(10) << "Width = "  << setw(10) << mWidth <<
                 setw(10) << "Rcut  = "  << setw(10) << mRcut  <<
                 endl;
    }

std::string
Gaussian::getName()
    {
    return "Gaussian";
    }

GReal
Gaussian::getScale()
    {
    return mScale;
    }

GReal
Gaussian::getWidth()
    {
    return mWidth;
    }

GReal
Gaussian::getRcut()
    {
    return mRcut;
    }

template <>
PairEnergy::PairEnergyPtr
mixPairEnergy<Gaussian>(PairEnergy& e1, PairEnergy& e2)
    {
    Gaussian* p1 = nullptr;
    Gaussian* p2 = nullptr;

    p1 = dynamic_cast<Gaussian*>(&e1);
    p2 = dynamic_cast<Gaussian*>(&e2);

    if (p1 == nullptr or p2 == nullptr)
        THROW_EXCEPT("Invalid mixing occurs");

    double scale  = std::sqrt(p1->getScale() * p2->getScale());
    double width  = 0.5 * (p1->getWidth()  + p2->getWidth());
    double rcut   = 0.5 * (p1->getRcut() + p2->getRcut());

    return std::make_unique<Gaussian>(scale, width, rcut);
    }
