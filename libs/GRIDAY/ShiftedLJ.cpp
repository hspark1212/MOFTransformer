#include "ShiftedLJ.hpp"

#include <iostream>
#include <iomanip>
#include <limits>
#include <cmath>


#include "GridayException.hpp"

ShiftedLJ::ShiftedLJ(GReal eps, GReal sig, GReal rcut) :
    PairEnergy {},
    mEps {eps},
    mSig {sig},
    mSigSq {sig * sig},
    mRcut {rcut},
    mRcutSq {rcut * rcut}
    {

    }

void
ShiftedLJ::setSimulationBox(const Cell& box)
    {
    mBox = box;
    mInvBox = inverse(box);
    }

GReal
ShiftedLJ::calculate(const Vectors& r1, const Vectors& r2)
    {
    GReal zero = static_cast<GReal>(0.001);

    bool cond1 = mEps  < zero;
    bool cond2 = mSig  < zero;
    bool cond3 = mRcut < zero;

    if (cond1 || cond2 || cond3)
        return static_cast<GReal>(0.0);

    if (r1.size() == 0 || r2.size() == 0)
        return static_cast<GReal>(0.0);

    int size1 = r1.size();
    int size2 = r2.size();

    GReal overlapDistSq = 1e-8;
    GReal one = 1.0;
    GReal energy = 0.0;

    // Shifted cutoff
    GReal eCutOne = 2500.0;
    GReal eCutTwo = 20000.0;

    GReal eCutOneDis = static_cast<GReal>(1 + std::sqrt(eCutOne / mEps + 1));
    eCutOneDis /= 2.0;
    GReal power = -1.0/6.0;
    eCutOneDis = static_cast<GReal>(mSig * std::pow(eCutOneDis, power));
    //std::cout << "Ecut dis " << eCutOneDis << std::endl;

    // matching f' at eCutOne
    /*
    GReal diff = static_cast<GReal>(std::sqrt((mEps + eCutOne) / mEps));
    GReal eCutDiff = static_cast<GReal>(std::pow(1.0 + diff, 7/6.0));
    eCutDiff *= diff;
    eCutDiff *= static_cast<GReal>(std::pow(2, 5/6.0));
    eCutDiff = eCutDiff * -6.0 * mEps / mSig;
    std::cout << "Ecut diff " << eCutDiff << std::endl;
    */


    // 1. Quadratic
    GReal shiftA = eCutOne - eCutTwo;
    shiftA = shiftA / eCutOneDis / eCutOneDis;
    GReal shiftC = eCutTwo;
    shiftA = shiftA / (4.0 * mEps);
    shiftC = shiftC / (4.0 * mEps);

    // 2. Linear
    GReal linearA = (eCutOne - eCutTwo) / eCutOneDis;
    GReal linearB = eCutTwo;
    linearA = linearA / (4.0 * mEps);
    linearB = linearB / (4.0 * mEps);

    GReal eCutOneDisSq = eCutOneDis * eCutOneDis;

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
                auto r2 = mSigSq / rsq;
                auto r6 = r2 * r2 * r2;

                // 1. Quadratic
                if (rsq < eCutOneDisSq)
                    {
                    energy += shiftA * rsq + shiftC;
                    }
                else
                    {
                    energy += r6 * (r6 - one);
                    }

                // 2. Linear
                /*
                if (rsq < eCutOneDisSq)
                    {
                    energy += linearA * static_cast<GReal>(std::sqrt(rsq)) + linearB;
                    }
                else
                    {
                    energy += r6 * (r6 - one);
                    }
                */
                /*
                if (rsq < 1.0)
                    {
                    energy += eCutTwo / (4.0 * mEps);
                    }
                else if (rsq < eCutOneDisSq)
                    {
                    energy += eCutOne / (4.0 * mEps);
                    }
                else
                    {
                    energy += r6 * (r6 - one);
                    }
                */
                }
            }
        }

    energy *= 4.0 * mEps;

    return energy;
    }

std::unique_ptr<PairEnergy>
ShiftedLJ::clone()
    {
    return std::make_unique<ShiftedLJ>(mEps, mSig, mRcut);
    }

void
ShiftedLJ::print()
    {
    using namespace std;

    cout << "Pair Type: ShiftedLJ, Parameter: " <<
                 setw(10) << "Eps = "  << setw(10) << mEps <<
                 setw(10) << "Sig = "  << setw(10) << mSig <<
                 setw(10) << "Rcut = " << setw(10) << mRcut <<
                 endl;
    }

std::string
ShiftedLJ::getName()
    {
    return "ShiftedLJ";
    }

GReal
ShiftedLJ::getEps()
    {
    return mEps;
    }

GReal
ShiftedLJ::getSig()
    {
    return mSig;
    }

GReal
ShiftedLJ::getRcut()
    {
    return mRcut;
    }

template <>
PairEnergy::PairEnergyPtr
mixPairEnergy<ShiftedLJ>(PairEnergy& e1, PairEnergy& e2)
    {
    ShiftedLJ* p1 = nullptr;
    ShiftedLJ* p2 = nullptr;

    p1 = dynamic_cast<ShiftedLJ*>(&e1);
    p2 = dynamic_cast<ShiftedLJ*>(&e2);

    if (p1 == nullptr or p2 == nullptr)
        THROW_EXCEPT("Invalid mixing occurs");

    double eps  = std::sqrt(p1->getEps() * p2->getEps());
    double sig  = 0.5 * (p1->getSig()  + p2->getSig());
    double rcut = 0.5 * (p1->getRcut() + p2->getRcut());

    return std::make_unique<ShiftedLJ>(eps, sig, rcut);
    }
