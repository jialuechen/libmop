
#ifndef libmop_MEAN_FUNCTION_ARD_HPP
#define libmop_MEAN_FUNCTION_ARD_HPP

#include <libmop/mean/mean.hpp>

namespace libmop {
    namespace mean {

        /// Functor used to optimize the mean function using the maximum likelihood principle
        ///
        /// It incorporates the hyperparameters of the underlying mean function, if any
        /// @see libmop::model::gp::KernelMeanLFOpt, libmop::model::gp::MeanLFOpt
        template <typename Params, typename MeanFunction>
        struct FunctionARD : public BaseMean<Params> {
            FunctionARD(size_t dim_out = 1)
                : _mean_function(dim_out), _tr(dim_out, dim_out + 1)
            {
                Eigen::VectorXd h = Eigen::VectorXd::Zero(dim_out * (dim_out + 1) + _mean_function.h_params_size());
                for (size_t i = 0; i < dim_out; i++)
                    h[i * (dim_out + 2)] = 1;
                if (_mean_function.h_params_size() > 0)
                    h.tail(_mean_function.h_params_size()) = _mean_function.h_params();
                this->set_h_params(h);
            }

            size_t h_params_size() const { return _tr.rows() * _tr.cols() + _mean_function.h_params_size(); }

            Eigen::VectorXd h_params() const
            {
                Eigen::VectorXd params(h_params_size());
                params.head(_tr.rows() * _tr.cols()) = _h_params;
                if (_mean_function.h_params_size() > 0)
                    params.tail(_mean_function.h_params_size()) = _mean_function.h_params();

                return params;
            }

            void set_h_params(const Eigen::VectorXd& p)
            {
                _h_params = p.head(_tr.rows() * _tr.cols());
                for (int c = 0; c < _tr.cols(); c++)
                    for (int r = 0; r < _tr.rows(); r++)
                        _tr(r, c) = p[r * _tr.cols() + c];

                if (_mean_function.h_params_size() > 0)
                    _mean_function.set_h_params(p.tail(_mean_function.h_params_size()));
            }

            template <typename GP>
            Eigen::MatrixXd grad(const Eigen::VectorXd& x, const GP& gp) const
            {
                Eigen::MatrixXd grad = Eigen::MatrixXd::Zero(_tr.rows(), h_params_size());
                Eigen::VectorXd m = _mean_function(x, gp);
                for (int i = 0; i < _tr.rows(); i++) {
                    grad.block(i, i * _tr.cols(), 1, _tr.cols() - 1) = m.transpose();
                    grad(i, (i + 1) * _tr.cols() - 1) = 1;
                }
                if (_mean_function.h_params_size() > 0) {
                    Eigen::MatrixXd m_grad = Eigen::MatrixXd::Zero(_tr.rows() + 1, _mean_function.h_params_size());
                    m_grad.block(0, 0, _tr.rows(), _mean_function.h_params_size()) = _mean_function.grad(x, gp);
                    Eigen::MatrixXd gg = _tr * m_grad;
                    grad.block(0, h_params_size() - _mean_function.h_params_size(), _tr.rows(), _mean_function.h_params_size()) = gg;
                }
                return grad;
            }

            template <typename GP>
            Eigen::VectorXd operator()(const Eigen::VectorXd& x, const GP& gp) const
            {
                Eigen::VectorXd m = _mean_function(x, gp);
                Eigen::VectorXd m_1(m.size() + 1);
                m_1.head(m.size()) = m;
                m_1[m.size()] = 1;
                return _tr * m_1;
            }

        protected:
            MeanFunction _mean_function;
            Eigen::MatrixXd _tr;
            Eigen::VectorXd _h_params;
        };
    } // namespace mean
} // namespace libmop

#endif
