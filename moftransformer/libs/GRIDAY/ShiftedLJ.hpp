#pragma once

#include "PairEnergy.hpp"

class ShiftedLJ : public PairEnergy
    {
public:
    ShiftedLJ(GReal eps, GReal sig, GReal rcut);
    virtual ~ShiftedLJ() = default;

    virtual void setSimulationBox(const Cell& box);
    virtual GReal calculate(const Vectors& r1, const Vectors& r2);

    virtual std::unique_ptr<PairEnergy> clone();
    virtual void print();
    virtual std::string getName();

    GReal getEps();
    GReal getSig();
    virtual GReal getRcut();

private:
    GReal mEps;
    GReal mSig;
    GReal mSigSq;
    GReal mRcut;
    GReal mRcutSq;
    Cell mBox;
    Cell mInvBox;
    };

template <>
PairEnergy::PairEnergyPtr mixPairEnergy<ShiftedLJ>(PairEnergy& e1,
                                                      PairEnergy& e2);
