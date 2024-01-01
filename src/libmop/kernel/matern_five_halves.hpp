
#ifndef libmop_KERNEL_MATERN_FIVE_HALVES_HPP
#define libmop_KERNEL_MATERN_FIVE_HALVES_HPP

#include <libmop/kernel/kernel.hpp>

namespace libmop {
    namespace defaults {
        struct kernel_maternfivehalves {
            /// @ingroup kernel_defaults
            BO_PARAM(double, sigma_sq, 1);
            /// @ingroup kernel_defaults
            BO_PARAM(double, l, 1);
        };
    } // namespace defaults
    namespace kernel {
        /**
          @ingroup kernel

          \rst

          Matern kernel

          .. math::
            d = ||v1 - v2||

            \nu = 5/2

            C(d) = \sigma^2\frac{2^{1-\nu}}{\Gamma(\nu)}\Bigg(\sqrt{2\nu}\frac{d}{l}\Bigg)^\nu K_\nu\Bigg(\sqrt{2\nu}\frac{d}{l}\Bigg),


          Parameters:
            - ``double sigma_sq`` (signal variance)
            - ``double l`` (characteristic length scale)

          Reference: :cite:`matern1960spatial` & :cite:`brochu2010tutorial` p.10 & https://en.wikipedia.org/wiki/Mat%C3%A9rn_covariance_function
          \endrst
        */
        template <typename Params>
        struct MaternFiveHalves : public BaseKernel<Params, MaternFiveHalves<Params>> {
            MaternFiveHalves(size_t dim = 1) : _sf2(Params::kernel_maternfivehalves::sigma_sq()), _l(Params::kernel_maternfivehalves::l())
            {
                _h_params = Eigen::VectorXd(2);
                _h_params << std::log(_l), std::log(std::sqrt(_sf2));
            }

            size_t params_size() const { return 2; }

            // Return the hyper parameters in log-space
            Eigen::VectorXd params() const { return _h_params; }

            // We expect the input parameters to be in log-space
            void set_params(const Eigen::VectorXd& p)
            {
                _h_params = p;
                _l = std::exp(p(0));
                _sf2 = std::exp(2.0 * p(1));
            }

            double kernel(const Eigen::VectorXd& v1, const Eigen::VectorXd& v2) const
            {
                double d = (v1 - v2).norm();
                double d_sq = d * d;
                double l_sq = _l * _l;
                double term1 = std::sqrt(5) * d / _l;
                double term2 = 5. * d_sq / (3. * l_sq);

                return _sf2 * (1 + term1 + term2) * std::exp(-term1);
            }

            Eigen::VectorXd gradient(const Eigen::VectorXd& x1, const Eigen::VectorXd& x2) const
            {
                Eigen::VectorXd grad(this->params_size());

                double d = (x1 - x2).norm();
                double d_sq = d * d;
                double l_sq = _l * _l;
                double term1 = std::sqrt(5) * d / _l;
                double term2 = 5. * d_sq / (3. * l_sq);
                double r = std::exp(-term1);

                // derivative of term1 = -term1
                // derivative of term2 = -2*term2
                // derivative of e^(-term1) = term1*r
                grad(0) = _sf2 * (r * term1 * (1 + term1 + term2) + (-term1 - 2. * term2) * r);
                grad(1) = 2 * _sf2 * (1 + term1 + term2) * r;

                return grad;
            }

        protected:
            double _sf2, _l;

            Eigen::VectorXd _h_params;
        };
    } // namespace kernel
} // namespace libmop

#endif
