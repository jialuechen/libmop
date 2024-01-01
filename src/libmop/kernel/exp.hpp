
#ifndef libmop_KERNEL_EXP_HPP
#define libmop_KERNEL_EXP_HPP

#include <libmop/kernel/kernel.hpp>

namespace libmop {
    namespace defaults {
        struct kernel_exp {
            /// @ingroup kernel_defaults
            BO_PARAM(double, sigma_sq, 1);
            BO_PARAM(double, l, 1);
        };
    } // namespace defaults
    namespace kernel {
        /**
          @ingroup kernel
          \rst
          Exponential kernel (see :cite:`brochu2010tutorial` p. 9).

          .. math::
              k(v_1, v_2)  = \sigma^2\exp \Big(-\frac{||v_1 - v_2||^2}{2l^2}\Big)

          Parameters:
            - ``double sigma_sq`` (signal variance)
            - ``double l`` (characteristic length scale)
          \endrst
        */
        template <typename Params>
        struct Exp : public BaseKernel<Params, Exp<Params>> {
            Exp(size_t dim = 1) : _sf2(Params::kernel_exp::sigma_sq()), _l(Params::kernel_exp::l())
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
                double l_sq = _l * _l;
                double r = (v1 - v2).squaredNorm() / l_sq;
                return _sf2 * std::exp(-0.5 * r);
            }

            Eigen::VectorXd gradient(const Eigen::VectorXd& x1, const Eigen::VectorXd& x2) const
            {
                Eigen::VectorXd grad(this->params_size());
                double l_sq = _l * _l;
                double r = (x1 - x2).squaredNorm() / l_sq;
                double k = _sf2 * std::exp(-0.5 * r);

                grad(0) = r * k;
                grad(1) = 2 * k;
                return grad;
            }

        protected:
            double _sf2, _l;

            Eigen::VectorXd _h_params;
        };
    } // namespace kernel
} // namespace libmop

#endif
