
#ifndef libmop_OPT_RANDOM_POINT_HPP
#define libmop_OPT_RANDOM_POINT_HPP

#include <Eigen/Core>
#include <libmop/tools/random_generator.hpp>

namespace libmop {
    namespace opt {
        /// @ingroup opt
        /// - return a random point in [0, 1]
        /// - no parameters
        /// - useful for control experiments (do not use this otherwise!)
        template <typename Params>
        struct RandomPoint {
            template <typename F>
            Eigen::VectorXd operator()(const F& f, const Eigen::VectorXd& init, bool bounded) const
            {
                // Random point does not support unbounded search
                assert(bounded);
                return tools::random_vector(init.size());
            }
        };
    }
}

#endif
