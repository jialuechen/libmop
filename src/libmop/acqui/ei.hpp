#ifndef libmop_ACQUI_EI_HPP
#define libmop_ACQUI_EI_HPP

#include <Eigen/Core>
#include <cmath>
#include <vector>

#include <libmop/opt/optimizer.hpp>
#include <libmop/tools/macros.hpp>

namespace libmop {
    namespace defaults {
        struct acqui_ei {
            /// @ingroup acqui_defaults
            BO_PARAM(double, jitter, 0.0);
        };
    }
    namespace acqui {
        /** @ingroup acqui
        \rst
        Classic EI (Expected Improvement). See :cite:`brochu2010tutorial`, p. 14

          .. math::
            EI(x) = (\mu(x) - f(x^+) - \xi)\Phi(Z) + \sigma(x)\phi(Z),\\\text{with } Z = \frac{\mu(x)-f(x^+) - \xi}{\sigma(x)}.

        Parameters:
          - ``double jitter`` - :math:`\xi`
        \endrst
        */
        template <typename Params, typename Model>
        class EI {
        public:
            EI(const Model& model, int iteration = 0) : _model(model), _nb_samples(-1) {}

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

                // Compute EI(x)
                // First find the best so far (predicted) observation -- if needed
                if (_nb_samples != _model.nb_samples()) {
                    std::vector<double> rewards;
                    for (auto s : _model.samples()) {
                        rewards.push_back(afun(_model.mu(s)));
                    }

                    _nb_samples = _model.nb_samples();
                    _f_max = *std::max_element(rewards.begin(), rewards.end());
                }
                // Calculate Z and \Phi(Z) and \phi(Z)
                double X = afun(mu) - _f_max - Params::acqui_ei::jitter();
                double Z = X / sigma;
                double phi = std::exp(-0.5 * std::pow(Z, 2.0)) / std::sqrt(2.0 * M_PI);
                double Phi = 0.5 * std::erfc(-Z / std::sqrt(2)); //0.5 * (1.0 + std::erf(Z / std::sqrt(2)));

                return opt::no_grad(X * Phi + sigma * phi);
            }

        protected:
            const Model& _model;
            int _nb_samples;
            double _f_max;
        };
    }
}

#endif
