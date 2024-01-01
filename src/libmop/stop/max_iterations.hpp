
#ifndef libmop_STOP_MAX_ITERATIONS_HPP
#define libmop_STOP_MAX_ITERATIONS_HPP

#include <libmop/tools/macros.hpp>

namespace libmop {
    namespace defaults {
        struct stop_maxiterations {
            /// @ingroup stop_defaults
            BO_PARAM(int, iterations, 190);
        };
    }
    namespace stop {
        /// @ingroup stop
        /// Stop after a given number of iterations
        ///
        /// parameter: int iterations
        template <typename Params>
        struct MaxIterations {
            MaxIterations() {}

            template <typename BO, typename AggregatorFunction>
            bool operator()(const BO& bo, const AggregatorFunction&)
            {
                return bo.current_iteration() >= Params::stop_maxiterations::iterations();
            }
        };
    }
}

#endif
