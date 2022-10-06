#pragma once

#include "PairEnergy.hpp"

class Gaussian : public PairEnergy
    {
public:
    Gaussian(GReal scale, GReal width, GReal rcut);
    virtual ~Gaussian() = default;

    virtual void setSimulationBox(const Cell& box);
    virtual GReal calculate(const Vectors& r1, const Vectors& r2);

    virtual std::unique_ptr<PairEnergy> clone();
    virtual void print();
    virtual std::string getName();

    GReal getScale();
    GReal getWidth();
    virtual GReal getRcut();

private:
    GReal mScale;
    GReal mWidth;
    //GReal mScaleSq;
    GReal mRcut;
    GReal mRcutSq;
    Cell mBox;
    Cell mInvBox;
    };

template <>
PairEnergy::PairEnergyPtr mixPairEnergy<Gaussian>(PairEnergy& e1,
                                                      PairEnergy& e2);
