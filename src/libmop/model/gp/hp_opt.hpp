
#ifndef libmop_MODEL_GP_HP_OPT_HPP
#define libmop_MODEL_GP_HP_OPT_HPP

#include <Eigen/Core>

#include <libmop/opt/rprop.hpp>

namespace libmop {
    namespace model {
        namespace gp {
            ///@ingroup model_opt
            ///base class for optimization of the hyper-parameters of a GP
            template <typename Params, typename Optimizer = opt::Rprop<Params>>
            struct HPOpt {
            public:
                HPOpt() : _called(false) {}
                /// to avoid stupid warnings
                HPOpt(const HPOpt&) { _called = true; }
                ~HPOpt()
                {
                    if (!_called) {
                        std::cerr << "'HPOpt' was never called!" << std::endl;
                    }
                }

            protected:
                bool _called;
            };
        } // namespace gp
    } // namespace model
} // namespace libmop

#endif
