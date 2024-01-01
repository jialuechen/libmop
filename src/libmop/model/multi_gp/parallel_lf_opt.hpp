
#ifndef libmop_MODEL_MULTI_GP_PARALLEL_LF_OPT_HPP
#define libmop_MODEL_MULTI_GP_PARALLEL_LF_OPT_HPP

#include <libmop/model/gp/hp_opt.hpp>

namespace libmop {
    namespace model {
        namespace multi_gp {
            ///@ingroup model_opt
            ///optimize each GP independently in parallel using HyperParamsOptimizer
            template <typename Params, typename HyperParamsOptimizer = libmop::model::gp::NoLFOpt<Params>>
            struct ParallelLFOpt : public libmop::model::gp::HPOpt<Params> {
            public:
                template <typename GP>
                void operator()(GP& gp)
                {
                    this->_called = true;
                    auto& gps = gp.gp_models();
                    libmop::tools::par::loop(0, gps.size(), [&](size_t i) {
                        HyperParamsOptimizer hp_optimize;
                        hp_optimize(gps[i]);
                    });
                }
            };

        } // namespace multi_gp
    } // namespace model
} // namespace libmop

#endif