
#ifndef libmop_MODEL_GP_KERNEL_MEAN_LF_OPT_HPP
#define libmop_MODEL_GP_KERNEL_MEAN_LF_OPT_HPP

#include <libmop/model/gp/hp_opt.hpp>

namespace libmop {
    namespace model {
        namespace gp {
            ///@ingroup model_opt
            ///optimize the likelihood of both the kernel and the mean (try to align the mean function)
            template <typename Params, typename Optimizer = opt::Rprop<Params>>
            struct KernelMeanLFOpt : public HPOpt<Params, Optimizer> {
            public:
                template <typename GP>
                void operator()(GP& gp)
                {
                    this->_called = true;
                    KernelMeanLFOptimization<GP> optimization(gp);
                    Optimizer optimizer;

                    int dim = gp.kernel_function().h_params_size() + gp.mean_function().h_params_size();

                    Eigen::VectorXd init(dim);
                    init.head(gp.kernel_function().h_params_size()) = gp.kernel_function().h_params();
                    init.tail(gp.mean_function().h_params_size()) = gp.mean_function().h_params();

                    Eigen::VectorXd params = optimizer(optimization, init, false);
                    gp.kernel_function().set_h_params(params.head(gp.kernel_function().h_params_size()));
                    gp.mean_function().set_h_params(params.tail(gp.mean_function().h_params_size()));

                    gp.recompute(true);
                    gp.compute_log_lik();
                }

            protected:
                template <typename GP>
                struct KernelMeanLFOptimization {
                public:
                    KernelMeanLFOptimization(const GP& gp) : _original_gp(gp) {}

                    opt::eval_t operator()(const Eigen::VectorXd& params, bool compute_grad) const
                    {
                        GP gp(this->_original_gp);
                        gp.kernel_function().set_h_params(params.head(gp.kernel_function().h_params_size()));
                        gp.mean_function().set_h_params(params.tail(gp.mean_function().h_params_size()));

                        gp.recompute(true);

                        double lik = gp.compute_log_lik();

                        if (!compute_grad)
                            return opt::no_grad(lik);

                        Eigen::VectorXd grad = Eigen::VectorXd::Zero(params.size());

                        grad.head(gp.kernel_function().h_params_size()) = gp.compute_kernel_grad_log_lik();
                        grad.tail(gp.mean_function().h_params_size()) = gp.compute_mean_grad_log_lik();

                        return {lik, grad};
                    }

                protected:
                    const GP& _original_gp;
                };
            };
        } // namespace gp
    } // namespace model
} // namespace libmop

#endif
