
#ifndef libmop_MODEL_GP_MEAN_LF_OPT_HPP
#define libmop_MODEL_GP_MEAN_LF_OPT_HPP

#include <libmop/model/gp/hp_opt.hpp>

namespace libmop {
    namespace model {
        namespace gp {
            ///@ingroup model_opt
            ///optimize the likelihood of the mean only (try to align the mean function)
            template <typename Params, typename Optimizer = opt::Rprop<Params>>
            struct MeanLFOpt : public HPOpt<Params, Optimizer> {
            public:
                template <typename GP>
                void operator()(GP& gp)
                {
                    this->_called = true;
                    MeanLFOptimization<GP> optimization(gp);
                    Optimizer optimizer;
                    Eigen::VectorXd params = optimizer(optimization, gp.mean_function().h_params(), false);
                    gp.mean_function().set_h_params(params);
                    gp.recompute(true, false);
                    gp.compute_log_lik();
                }

            protected:
                template <typename GP>
                struct MeanLFOptimization {
                public:
                    MeanLFOptimization(const GP& gp) : _original_gp(gp)
                    {
                        _original_gp.compute_inv_kernel();
                    }

                    opt::eval_t operator()(const Eigen::VectorXd& params, bool compute_grad) const
                    {
                        GP gp(this->_original_gp);
                        gp.mean_function().set_h_params(params);

                        gp.recompute(true, false);

                        double lik = gp.compute_log_lik();

                        if (!compute_grad)
                            return opt::no_grad(lik);

                        Eigen::VectorXd grad = gp.compute_mean_grad_log_lik();

                        return {lik, grad};
                    }

                protected:
                    GP _original_gp;
                };
            };
        } // namespace gp
    } // namespace model
} // namespace libmop

#endif
