
#ifndef libmop_ACQUI_ECI_HPP
#define libmop_ACQUI_ECI_HPP

#include <Eigen/Core>
#include <cmath>
#include <vector>

#include <libmop/tools/macros.hpp>

namespace libmop {
    namespace defaults {
        struct acqui_eci {
            /// @ingroup acqui_defaults
            BO_PARAM(double, jitter, 0.0);
        };
    }

    namespace experimental {
        namespace acqui {
            template <typename Params, typename Model, typename ConstraintModel>
            class ECI {
            public:
                ECI(const Model& model, const ConstraintModel& constraint_model, int iteration = 0)
                    : _model(model), _constraint_model(constraint_model), _nb_samples(-1) {}

                size_t dim_in() const { return _model.dim_in(); }

                size_t dim_out() const { return _model.dim_out(); }

                template <typename AggregatorFunction>
                opt::eval_t operator()(const Eigen::VectorXd& v, const AggregatorFunction& afun, bool gradient)
                {
                    assert(!gradient);

                    Eigen::VectorXd mu;
                    double sigma_sq;
                    std::tie(mu, sigma_sq) = _model.query(v);
                    double sigma = std::sqrt(sigma_sq);

                    // If \sigma(x) = 0 or we do not have any observation yet we return 0
                    if (sigma < 1e-10 || _model.samples().size() < 1)
                        return opt::no_grad(0.0);

                    // Compute expected constrained improvement
                    // First find the best (predicted) observation so far -- if needed
                    if (_nb_samples != _model.nb_samples()) {
                        std::vector<double> rewards;
                        for (auto s : _model.samples()) {
                            rewards.push_back(afun(_model.mu(s)));
                        }

                        _nb_samples = _model.nb_samples();
                        _f_max = *std::max_element(rewards.begin(), rewards.end());
                    }
                    // Calculate Z and \Phi(Z) and \phi(Z)
                    double X = afun(mu) - _f_max - Params::acqui_eci::jitter();
                    double Z = X / sigma;
                    double phi = std::exp(-0.5 * std::pow(Z, 2.0)) / std::sqrt(2.0 * M_PI);
                    double Phi = 0.5 * std::erfc(-Z / std::sqrt(2));

                    return opt::no_grad(_pf(v, afun) * (X * Phi + sigma * phi));
                }

            protected:
                const Model& _model;
                const ConstraintModel& _constraint_model;
                int _nb_samples;
                double _f_max;

                template <typename AggregatorFunction>
                double _pf(const Eigen::VectorXd& v, const AggregatorFunction& afun) const
                {
                    Eigen::VectorXd mu;
                    double sigma_sq;
                    std::tie(mu, sigma_sq) = _constraint_model.query(v);
                    double sigma = std::sqrt(sigma_sq);

                    if (sigma < 1e-10 || _constraint_model.samples().size() < 1)
                        return 1.0;

                    double Z = (afun(mu) - 1.0) / sigma;
                    double Phi = 0.5 * std::erfc(-Z / std::sqrt(2));

                    return Phi;
                }
            };
        }
    }
}

#endif
