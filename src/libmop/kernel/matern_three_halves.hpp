#ifndef libmop_KERNEL_MATERN_THREE_HALVES_HPP
#define libmop_KERNEL_MATERN_THREE_HALVES_HPP

#include <libmop/kernel/kernel.hpp>

namespace libmop {
    namespace defaults {
        struct kernel_maternthreehalves {
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
         Matern 3/2 kernel

         .. math::
           d = ||v1 - v2||

           \nu = 3/2

           C(d) = \sigma^2\frac{2^{1-\nu}}{\Gamma(\nu)}\Bigg(\sqrt{2\nu}\frac{d}{l}\Bigg)^\nu K_\nu\Bigg(\sqrt{2\nu}\frac{d}{l}\Bigg),


         Parameters:
          - ``double sigma_sq`` (signal variance)
          - ``double l`` (characteristic length scale)

        Reference: :cite:`matern1960spatial` & :cite:`brochu2010tutorial` p.10 & https://en.wikipedia.org/wiki/Mat%C3%A9rn_covariance_function
        \endrst
        */
        template <typename Params>
        struct MaternThreeHalves : public BaseKernel<Params, MaternThreeHalves<Params>> {
            MaternThreeHalves(size_t dim = 1) : _sf2(Params::kernel_maternthreehalves::sigma_sq()), _l(Params::kernel_maternthreehalves::l())
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
                double term = std::sqrt(3) * d / _l;

                return _sf2 * (1 + term) * std::exp(-term);
            }

            Eigen::VectorXd gradient(const Eigen::VectorXd& x1, const Eigen::VectorXd& x2) const
            {
                Eigen::VectorXd grad(this->params_size());

                double d = (x1 - x2).norm();
                double term = std::sqrt(3) * d / _l;
                double r = std::exp(-term);

                // derivative of (1+term) = -term
                // derivative of e^(-term) = term*r
                grad(0) = _sf2 * (-term * r + (1 + term) * term * r);
                grad(1) = 2 * _sf2 * (1 + term) * r;

                return grad;
            }

        protected:
            double _sf2, _l;

            Eigen::VectorXd _h_params;
        };
    } // namespace kernel
} // namespace libmop

#endif
