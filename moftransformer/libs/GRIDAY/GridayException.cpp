#include "GridayException.hpp"

#include <sstream>

GridayException::GridayException(const std::string& file, int line,
                                 const std::string& msg)
    : mDepth {0}
    {
    std::stringstream ss;

    ss << "File: " << file << ", Line: " << line << ", Message: " << msg;
    mMsg = ss.str();
    }

GridayException::GridayException(const std::string& file, int line,
                                 const std::string& msg,
                                 const GridayException& parent)
    : mDepth {parent.getDepth() + 1}
    {
    std::stringstream ss;

    ss << parent.what() << std::endl;

    for (int i = 0; i < mDepth; ++i)
        ss << "    ";

    ss << "File: " << file << ", Line: " << line << ", Message: " << msg;

    mMsg = ss.str();
    }

const char*
GridayException::what() const noexcept
    {
    return mMsg.c_str();
    }

int
GridayException::getDepth() const
    {
    return mDepth;
    }
