
#ifndef libmop_MODEL_GP_KERNEL_LF_OPT_HPP
#define libmop_MODEL_GP_KERNEL_LF_OPT_HPP

#include <libmop/model/gp/hp_opt.hpp>

namespace libmop {
    namespace model {
        namespace gp {
            ///@ingroup model_opt
            ///optimize the likelihood of the kernel only
            template <typename Params, typename Optimizer = opt::Rprop<Params>>
            struct KernelLFOpt : public HPOpt<Params, Optimizer> {
            public:
                template <typename GP>
                void operator()(GP& gp)
                {
                    this->_called = true;
                    KernelLFOptimization<GP> optimization(gp);
                    Optimizer optimizer;
                    Eigen::VectorXd params = optimizer(optimization, gp.kernel_function().h_params(), false);
                    gp.kernel_function().set_h_params(params);
                    gp.recompute(false);
                    gp.compute_log_lik();
                }

            protected:
                template <typename GP>
                struct KernelLFOptimization {
                public:
                    KernelLFOptimization(const GP& gp) : _original_gp(gp) {}

                    opt::eval_t operator()(const Eigen::VectorXd& params, bool compute_grad) const
                    {
                        GP gp(this->_original_gp);
                        gp.kernel_function().set_h_params(params);

                        gp.recompute(false);

                        double lik = gp.compute_log_lik();

                        if (!compute_grad)
                            return opt::no_grad(lik);

                        Eigen::VectorXd grad = gp.compute_kernel_grad_log_lik();

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
