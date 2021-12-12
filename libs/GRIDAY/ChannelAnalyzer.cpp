#include "ChannelAnalyzer.hpp"

//#include <set>
#include <unordered_set>
#include <stack>
#include <vector>
#include <array>
#include <limits>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <cstdint>

// Hash function for unordered_set
namespace std
{
    template<typename T, size_t N>
    struct hash< array<T, N> >
    {
        typedef array<T, N> argument_type;
        typedef size_t result_type;

        result_type operator()(const argument_type& a) const
        {
            hash<T> hasher;
            result_type h = 0;
            for (result_type i = 0; i < N; ++i)
            {
                h = h * 31 + hasher(a[i]);
            }
            return h;
        }
    };
}

ChannelAnalyzer::ChannelAnalyzer(EnergyGrid& energyGrid, GReal maxE, GReal dr) :
    mEnergyGrid {energyGrid}
    {
    this->analyze(maxE, dr);
    }

int
ChannelAnalyzer::getChannelDimension()
    {
    return mChannelDimension;
    }

void
ChannelAnalyzer::analyze(GReal maxE, GReal dr)
    {
    using namespace std;

    // Ref energy grid.
    // Will be changed to copy if needed.
    EnergyGrid& eGrid = mEnergyGrid;

    //bool isVerbose = true;
    bool isVerbose = false;

    mChannelDimension = 0;

    GIndex invalid = {GIndex {0} - 1};
    // 0: yz plane
    // 1: zx plane
    // 2: xy palne

    // Get minimum rectangle cell.
    Cell box = eGrid.getContainingBox();
    Cell invBox = inverse(box);

    vector<GIndex> maxNs {static_cast<GIndex>(ceil(box.a[0] / dr)),
                          static_cast<GIndex>(ceil(box.b[1] / dr)),
                          static_cast<GIndex>(ceil(box.c[2] / dr))};

    vector<GIndex3> minIndices = eGrid.getLocalMinimumIndices(maxE);

    auto pbc = [maxNs, invalid](GIndex3& idx)
        {
        for (int i = 0; i < 3; ++i)
            {
            if (idx[i] == maxNs[i])
                idx[i] = 0;
            if (idx[i] == invalid)
                idx[i] = maxNs[i] - 1;
            }
        };

    auto idx2pos = [maxNs, box](const GIndex3& idx)
        {
        Vector s {static_cast<GReal>(idx[0]) / maxNs[0],
                  static_cast<GReal>(idx[1]) / maxNs[1],
                  static_cast<GReal>(idx[2]) / maxNs[2]};

        return box * s;
        };

    // Reduce start points
    auto dist = [this](const GIndex3& n1, const GIndex3& n2)
        {
        Cell cell = mEnergyGrid.getCell();

        GReal maxNx = mEnergyGrid.getMaxNx();
        GReal maxNy = mEnergyGrid.getMaxNy();
        GReal maxNz = mEnergyGrid.getMaxNz();

        Vector s1 {static_cast<GReal>(n1[0]) / maxNx,
                   static_cast<GReal>(n1[1]) / maxNy,
                   static_cast<GReal>(n1[2]) / maxNz};


        Vector s2 {static_cast<GReal>(n2[0]) / maxNx,
                   static_cast<GReal>(n2[1]) / maxNy,
                   static_cast<GReal>(n2[2]) / maxNz};

        Vector s = s2 - s1;

        for (auto& si : s)
            {
            if (si < -0.5)
                si += 1.0;
            if (si >  0.5)
                si -= 1.0;
            }

        return norm(cell * s);
        };

    if (minIndices.size() >= 2)
        {
        unordered_set<GIndex> nearList;
        for (GIndex i = 0; i < minIndices.size() - 1; ++i)
            for (GIndex j = i + 1; j < minIndices.size(); ++j)
                {
                if (dist(minIndices[i], minIndices[j]) < 2.0)
                    nearList.insert(j);
                }

        vector<GIndex3> dummy;

        for (GIndex i = 0; i < minIndices.size(); ++i)
            if (nearList.count(i) == 0)
                dummy.push_back(minIndices[i]);

        minIndices = dummy;
        }
    else
        {
        minIndices.resize(1);
        minIndices[0] = mEnergyGrid.getMinimumEnergyIndex3();
        }

    if (isVerbose)
        cout << "# of startring point = " << minIndices.size() << endl;

    auto tri2rec = [this, maxNs, invBox](const GIndex3& tri)
        {
        GReal tiny = numeric_limits<GReal>::epsilon();

        Cell cell = mEnergyGrid.getCell();

        GReal maxNx = mEnergyGrid.getMaxNx();
        GReal maxNy = mEnergyGrid.getMaxNy();
        GReal maxNz = mEnergyGrid.getMaxNz();

        Vector s {static_cast<GReal>(tri[0]) / maxNx,
                  static_cast<GReal>(tri[1]) / maxNy,
                  static_cast<GReal>(tri[2]) / maxNz};

        Vector r = cell * s;

        s = invBox * r;

        for (auto& si : s)
            {
            if (si < 0.0)
                si += 1.0;
            if (si > 1.0)
                si -= 1.0;
            }

        GIndex rec0 = floor(s[0] * maxNs[0] + tiny);
        GIndex rec1 = floor(s[1] * maxNs[1] + tiny);
        GIndex rec2 = floor(s[2] * maxNs[2] + tiny);
        GIndex3 rec = {rec0, rec1, rec2};

        return rec;
        };

    vector<string> directionStr =
        {"X Direction", "Y Direction", "Z Direction"};

    for (GIndex direction : {0, 1, 2})
        {
        if (isVerbose)
            cout << directionStr[direction] << " Searching start" << endl;

    for (const auto& minIndex : minIndices)
        {
        GIndex3 start = tri2rec(minIndex);
        GIndex3 point = start;
        int64_t counter = 0;

        unordered_set<GIndex3> visited {point};
        stack<GIndex3> possibleWays;
        stack<int64_t> counterStack;

        bool channelFound = false;

        while (not channelFound)
            {
            vector<GIndex3> trials (6, point);

            // Impose directional priority
            // Up direction 5 = first (stack)
            trials[5][direction]++; pbc(trials[5]);
            trials[0][direction]--; pbc(trials[0]);

            GIndex idx = 1;
            for (GIndex i = 0; i < 3; ++i)
                {
                if (i == direction)
                    continue;

                trials[idx][i]++; pbc(trials[idx]); idx++;
                trials[idx][i]--; pbc(trials[idx]); idx++;
                }

            for (GIndex i = 0; i < 6; ++i)
                {
                auto& trial = trials[i];

                if (visited.count(trial) == 0)
                    if (eGrid.interpolate(idx2pos(trial)) < maxE)
                        {
                        possibleWays.push(trial);

                        if (i == 0)
                            counterStack.push(counter - 1);
                        else if (i == 5)
                            counterStack.push(counter + 1);
                        else
                            counterStack.push(counter);
                        }
                }

            if (possibleWays.empty())
                {
                if (isVerbose)
                    cout << "Nowhere to go" << endl;

                break;
                }
            else
                {
                point = possibleWays.top();
                possibleWays.pop();
                visited.insert(point);

                counter = counterStack.top();
                counterStack.pop();
                }

            if (isVerbose)
                cout << "Current position = " << idx2pos(point) <<
                        setw(20) << "Index = " <<
                        setw(6) << point[0] <<
                        setw(6) << point[1] <<
                        setw(6) << point[2] <<
                        setw(20) << "Counter = " << counter <<
                        endl;

            if (static_cast<GIndex>(counter) == maxNs[direction])
                {
                if (isVerbose)
                    cout << directionStr[direction] << " Found" << endl;
                channelFound = true;
                mChannelDimension++;
                break;
                }
            }

        if (channelFound)
            break;
        } // minIndex loop

        } // direction
    }
