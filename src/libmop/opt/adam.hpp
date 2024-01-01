
#ifndef libmop_OPT_ADAM_HPP
#define libmop_OPT_ADAM_HPP

#include <algorithm>

#include <Eigen/Core>

#include <libmop/opt/optimizer.hpp>
#include <libmop/tools/macros.hpp>
#include <libmop/tools/math.hpp>

namespace libmop {
    namespace defaults {
        struct opt_adam {
            /// @ingroup opt_defaults
            /// number of max iterations
            BO_PARAM(int, iterations, 300);

            /// @ingroup opt_defaults
            /// alpha - learning rate
            BO_PARAM(double, alpha, 0.001);

            /// @ingroup opt_defaults
            /// β1
            BO_PARAM(double, b1, 0.9);

            /// @ingroup opt_defaults
            /// β2
            BO_PARAM(double, b2, 0.999);

            /// @ingroup opt_defaults
            /// norm epsilon for stopping
            BO_PARAM(double, eps_stop, 0.0);
        };
    } // namespace defaults
    namespace opt {
        /// @ingroup opt
        /// Adam optimizer
        /// Equations from: http://ruder.io/optimizing-gradient-descent/index.html#gradientdescentoptimizationalgorithms
        /// (I changed a bit the notation; η to α)
        ///
        /// Parameters:
        /// - int iterations
        /// - double alpha
        /// - double b1
        /// - double b2
        /// - double eps_stop
        template <typename Params>
        struct Adam {
            template <typename F>
            Eigen::VectorXd operator()(const F& f, const Eigen::VectorXd& init, bool bounded) const
            {
                assert(Params::opt_adam::b1() >= 0. && Params::opt_adam::b1() < 1.);
                assert(Params::opt_adam::b2() >= 0. && Params::opt_adam::b2() < 1.);
                assert(Params::opt_adam::alpha() >= 0.);

                size_t param_dim = init.size();
                double b1 = Params::opt_adam::b1();
                double b2 = Params::opt_adam::b2();
                double b1_t = b1;
                double b2_t = b2;
                double alpha = Params::opt_adam::alpha();
                double stop = Params::opt_adam::eps_stop();
                double epsilon = 1e-8;

                Eigen::VectorXd m = Eigen::VectorXd::Zero(param_dim);
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

                for (int i = 0; i < Params::opt_adam::iterations(); ++i) {
                    Eigen::VectorXd prev_params = params;
                    auto perf = opt::eval_grad(f, params);

                    Eigen::VectorXd grad = opt::grad(perf);
                    m = b1 * m.array() + (1. - b1) * grad.array();
                    v = b2 * v.array() + (1. - b2) * grad.array().square();

                    Eigen::VectorXd m_hat = m.array() / (1. - b1_t);
                    Eigen::VectorXd v_hat = v.array() / (1. - b2_t);

                    params.array() += alpha * m_hat.array() / (v_hat.array().sqrt() + epsilon);

                    b1_t *= b1;
                    b2_t *= b2;

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
