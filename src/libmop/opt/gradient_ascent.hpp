
#ifndef libmop_OPT_GRADIENT_ASCENT_HPP
#define libmop_OPT_GRADIENT_ASCENT_HPP

#include <algorithm>

#include <Eigen/Core>

#include <libmop/opt/optimizer.hpp>
#include <libmop/tools/macros.hpp>
#include <libmop/tools/math.hpp>

namespace libmop {
    namespace defaults {
        struct opt_gradient_ascent {
            /// @ingroup opt_defaults
            /// number of max iterations
            BO_PARAM(int, iterations, 300);

            /// @ingroup opt_defaults
            /// alpha - learning rate
            BO_PARAM(double, alpha, 0.001);

            /// @ingroup opt_defaults
            /// gamma - for momentum
            BO_PARAM(double, gamma, 0.0);

            /// @ingroup opt_defaults
            /// nesterov momentum; turn on/off
            BO_PARAM(bool, nesterov, false);

            /// @ingroup opt_defaults
            /// norm epsilon for stopping
            BO_PARAM(double, eps_stop, 0.0);
        };
    } // namespace defaults
    namespace opt {
        /// @ingroup opt
        /// Gradient Ascent with or without momentum (Nesterov or simple)
        /// Equations from: http://ruder.io/optimizing-gradient-descent/index.html#gradientdescentoptimizationalgorithms
        /// (I changed a bit the notation; η to α)
        ///
        /// Parameters:
        /// - int iterations
        /// - double alpha
        /// - double gamma
        /// - bool nesterov
        /// - double eps_stop
        template <typename Params>
        struct GradientAscent {
            template <typename F>
            Eigen::VectorXd operator()(const F& f, const Eigen::VectorXd& init, bool bounded) const
            {
                assert(Params::opt_gradient_ascent::gamma() >= 0. && Params::opt_gradient_ascent::gamma() < 1.);
                assert(Params::opt_gradient_ascent::alpha() >= 0.);

                size_t param_dim = init.size();
                double gamma = Params::opt_gradient_ascent::gamma();
                double alpha = Params::opt_gradient_ascent::alpha();
                double stop = Params::opt_gradient_ascent::eps_stop();
                bool is_nesterov = Params::opt_gradient_ascent::nesterov();

                Eigen::VectorXd v = Eigen::VectorXd::Zero(param_dim);

                Eigen::VectorXd params = init;

                if (bounded) {
                    for (int j = 0; j < params.size(); j++) {
                        if (params(j) < 0)
                            params(j) = 0;
                        if (params(j) > 1)
                            params(j) = 1;
                    }
                }

                for (int i = 0; i < Params::opt_gradient_ascent::iterations(); ++i) {
                    Eigen::VectorXd prev_params = params;
                    Eigen::VectorXd query_params = params;
                    // if Nesterov momentum, change query parameters
                    if (is_nesterov) {
                        query_params.array() += gamma * v.array();

                        // make sure that the parameters are still in bounds, if needed
                        if (bounded) {
                            for (int j = 0; j < query_params.size(); j++) {
                                if (query_params(j) < 0)
                                    query_params(j) = 0;
                                if (query_params(j) > 1)
                                    query_params(j) = 1;
                            }
                        }
                    }
                    auto perf = opt::eval_grad(f, query_params);

                    Eigen::VectorXd grad = opt::grad(perf);
                    v = gamma * v.array() + alpha * grad.array();

                    params.array() += v.array();

                    if (bounded) {
                        for (int j = 0; j < params.size(); j++) {
                            if (params(j) < 0)
                                params(j) = 0;
                            if (params(j) > 1)
                                params(j) = 1;
                        }
                    }

                    if ((prev_params - params).norm() < stop)
                        break;
                }

                return params;
            }
        };
    } // namespace opt
} // namespace libmop

#endif
