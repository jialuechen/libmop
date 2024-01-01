
#ifndef libmop_INIT_LHS_HPP
#define libmop_INIT_LHS_HPP

#include <Eigen/Core>

#include <libmop/tools/macros.hpp>
#include <libmop/tools/random_generator.hpp>

namespace libmop {
    namespace defaults {
        struct init_lhs {
            ///@ingroup init_defaults
            BO_PARAM(int, samples, 10);
        };
    } // namespace defaults
    namespace init {
        /** @ingroup init
          \rst
          Latin Hypercube sampling in [0, 1]^n (LHS)

          Parameters:
            - ``int samples`` (total number of samples)
          \endrst
        */
        template <typename Params>
        struct LHS {
            template <typename StateFunction, typename AggregatorFunction, typename Opt>
            void operator()(const StateFunction& seval, const AggregatorFunction&, Opt& opt) const
            {
                assert(Params::bayes_opt_bobase::bounded());

                Eigen::MatrixXd H = tools::random_lhs(StateFunction::dim_in(), Params::init_lhs::samples());

                for (int i = 0; i < Params::init_lhs::samples(); i++) {
                    opt.eval_and_add(seval, H.row(i));
                }
            }
        };
    } // namespace init
} // namespace libmop

#endif
