

#ifndef libmop_TOOLS_MATH_HPP
#define libmop_TOOLS_MATH_HPP

#include <cmath>
#include <cstdlib>
#include <ctime>
#include <list>
#include <mutex>
#include <random>
#include <stdlib.h>
#include <utility>

namespace libmop {
    namespace tools {

        /// @ingroup tools
        /// make a 1-D vector from a double (useful when we need to return vectors)
        inline Eigen::VectorXd make_vector(double x)
        {
            Eigen::VectorXd res(1);
            res(0) = x;
            return res;
        }

        template <typename T>
        inline constexpr int signum(T x, std::false_type is_signed)
        {
            return T(0) < x;
        }

        template <typename T>
        inline constexpr int signum(T x, std::true_type is_signed)
        {
            return (T(0) < x) - (x < T(0));
        }

        /// @ingroup tools
        /// return -1 if x < 0;
        /// return 0 if x = 0;
        /// return 1 if x > 0.
        template <typename T>
        inline constexpr int signum(T x)
        {
            return signum(x, std::is_signed<T>());
        }

        /// @ingroup tools
        /// return true if v is nan (not a number) or infinity
        template <typename T, typename std::enable_if<std::is_arithmetic<T>::value, int>::type = 0>
        inline bool is_nan_or_inf(T v)
        {
            return std::isinf(v) || std::isnan(v);
        }

        /// @ingroup tools
        /// return true if v is nan (not a number) or infinity
        /// (const version)
        template <typename T, typename std::enable_if<!std::is_arithmetic<T>::value, int>::type = 0>
        inline bool is_nan_or_inf(const T& v)
        {
            for (int i = 0; i < v.size(); ++i)
                if (std::isinf(v(i)) || std::isnan(v(i)))
                    return true;
            return false;
        }
    }
}

#endif
