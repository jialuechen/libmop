
#ifndef libmop_MEAN_CONSTANT_HPP
#define libmop_MEAN_CONSTANT_HPP

#include <libmop/mean/mean.hpp>

namespace libmop {
    namespace defaults {
        struct mean_constant {
            ///@ingroup mean_defaults
            BO_PARAM(double, constant, 1);
        };
    } // namespace defaults

    namespace mean {
        /** @ingroup mean
          A constant mean (the traditionnal choice for Bayesian optimization)

          Parameter:
            - ``double constant`` (the value of the constant)
        */
        template <typename Params>
        struct Constant : public BaseMean<Params> {
            Constant(size_t dim_out = 1) : _dim_out(dim_out), _constant(Params::mean_constant::constant()) {}

            template <typename GP>
            Eigen::VectorXd operator()(const Eigen::VectorXd& v, const GP&) const
            {
                return Eigen::VectorXd::Constant(_dim_out, _constant);
            }

            template <typename GP>
            Eigen::MatrixXd grad(const Eigen::VectorXd& x, const GP& gp) const
            {
                return Eigen::MatrixXd::Ones(_dim_out, 1);
            }

            size_t h_params_size() const { return 1; }

            Eigen::VectorXd h_params() const
            {
                Eigen::VectorXd p(1);
                p << _constant;
                return p;
            }

            void set_h_params(const Eigen::VectorXd& p)
            {
                _constant = p(0);
            }

        protected:
            size_t _dim_out;
            double _constant;
        };
    } // namespace mean
} // namespace libmop

#endif
