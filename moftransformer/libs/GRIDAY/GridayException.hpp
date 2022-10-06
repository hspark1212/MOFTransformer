#pragma once

#include <exception>
#include <string>

class GridayException : public std::exception
    {
public:
    GridayException(const std::string& file,
                    int line,
                    const std::string& msg);

    GridayException(const std::string& file,
                    int line,
                    const std::string& msg,
                    const GridayException& parent);

    virtual ~GridayException() = default;

    virtual const char* what() const noexcept;

    int getDepth() const;

private:
    int mDepth;
    std::string mMsg;
    };

#define THROW_EXCEPT(...) \
    throw GridayException {__FILE__, __LINE__, __VA_ARGS__};
