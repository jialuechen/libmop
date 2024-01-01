
#ifndef libmop_MODEL_GP_KERNEL_LOO_OPT_HPP
#define libmop_MODEL_GP_KERNEL_LOO_OPT_HPP

#include <libmop/model/gp/hp_opt.hpp>

namespace libmop {
    namespace model {
        namespace gp {
            ///@ingroup model_opt
            ///optimize the likelihood of the kernel only
            template <typename Params, typename Optimizer = opt::Rprop<Params>>
            struct KernelLooOpt : public HPOpt<Params, Optimizer> {
            public:
                template <typename GP>
                void operator()(GP& gp)
                {
                    this->_called = true;
                    KernelLooOptimization<GP> optimization(gp);
                    Optimizer optimizer;
                    Eigen::VectorXd params = optimizer(optimization, gp.kernel_function().h_params(), false);
                    gp.kernel_function().set_h_params(params);
                    gp.recompute(false);
                    gp.compute_log_loo_cv();
                }

            protected:
                template <typename GP>
                struct KernelLooOptimization {
                public:
                    KernelLooOptimization(const GP& gp) : _original_gp(gp) {}

                    opt::eval_t operator()(const Eigen::VectorXd& params, bool compute_grad) const
                    {
                        GP gp(this->_original_gp);
                        gp.kernel_function().set_h_params(params);

                        gp.recompute(false);

                        double loo = gp.compute_log_loo_cv();

                        if (!compute_grad)
                            return opt::no_grad(loo);

                        Eigen::VectorXd grad = gp.compute_kernel_grad_log_loo_cv();

                        return {loo, grad};
                    }

                protected:
                    const GP& _original_gp;
                };
            };
        } // namespace gp
    } // namespace model
} // namespace libmop

#endif
