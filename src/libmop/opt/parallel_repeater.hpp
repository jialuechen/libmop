
#ifndef libmop_OPT_PARALLEL_REPEATER_HPP
#define libmop_OPT_PARALLEL_REPEATER_HPP

#include <algorithm>

#include <Eigen/Core>

#include <libmop/opt/optimizer.hpp>
#include <libmop/tools/macros.hpp>
#include <libmop/tools/parallel.hpp>
#include <libmop/tools/random_generator.hpp>

namespace libmop {
    namespace defaults {
        struct opt_parallelrepeater {
            /// @ingroup opt_defaults
            /// number of replicates
            BO_PARAM(int, repeats, 10);

            /// epsilon of deviation: init + [-epsilon,epsilon]
            BO_PARAM(double, epsilon, 1e-2);
        };
    }
    namespace opt {
        /// @ingroup opt
        /// Meta-optimizer: run the same algorithm in parallel many times from different init points and return the maximum found among all the replicates
        /// (useful for local algorithms)
        ///
        /// Parameters:
        /// - int repeats
        template <typename Params, typename Optimizer>
        struct ParallelRepeater {
            template <typename F>
            Eigen::VectorXd operator()(const F& f, const Eigen::VectorXd& init, bool bounded) const
            {
                assert(Params::opt_parallelrepeater::repeats() > 0);
                assert(Params::opt_parallelrepeater::epsilon() > 0.);
                tools::par::init();
                using pair_t = std::pair<Eigen::VectorXd, double>;

                auto body = [&](int i) {
                    // clang-format off
                    Eigen::VectorXd r_deviation = tools::random_vector(init.size()).array() * 2. * Params::opt_parallelrepeater::epsilon() - Params::opt_parallelrepeater::epsilon();
                    Eigen::VectorXd v = Optimizer()(f, init + r_deviation, bounded);
                    double val = opt::eval(f, v);

                    return std::make_pair(v, val);
                    // clang-format on
                };

                auto comp = [](const pair_t& v1, const pair_t& v2) {
                    // clang-format off
                    return v1.second > v2.second;
                    // clang-format on
                };

                pair_t init_v = std::make_pair(init, -std::numeric_limits<float>::max());
                auto m = tools::par::max(init_v, Params::opt_parallelrepeater::repeats(), body, comp);

                return m.first;
            };
        };
    }
}

#endif
