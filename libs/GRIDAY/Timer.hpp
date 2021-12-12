#pragma once

#include <chrono>
#include <cstdint>

class Timer
    {
public:
     Timer();
    ~Timer();

    inline void   tic();
    inline Timer& toc();

    template <typename T = std::chrono::milliseconds>
    inline int64_t getDuration();
private:
    std::chrono::system_clock::time_point mStartTime;
    std::chrono::system_clock::time_point mEndTime;
    };

Timer::Timer()
    {
    this->tic();
    }

Timer::~Timer()
    {

    }

inline
void
Timer::tic()
    {
    mStartTime = std::chrono::system_clock::now();
    mEndTime = mStartTime;
    }

inline
Timer&
Timer::toc()
    {
    mEndTime = std::chrono::system_clock::now();

    return *this;
    }

template <typename T>
inline
int64_t
Timer::getDuration()
    {
    return std::chrono::duration_cast<T>(mEndTime - mStartTime).count();
    }
