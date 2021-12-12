#pragma once

#include "PairEnergy.hpp"

class LennardJones : public PairEnergy
    {
public:
    LennardJones(GReal eps, GReal sig, GReal rcut);
    virtual ~LennardJones() = default;

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
PairEnergy::PairEnergyPtr mixPairEnergy<LennardJones>(PairEnergy& e1,
                                                      PairEnergy& e2);
