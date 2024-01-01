
#ifndef libmop_MODEL_GP_NO_LF_OPT_HPP
#define libmop_MODEL_GP_NO_LF_OPT_HPP

#include <iostream>

namespace libmop {
    namespace model {
        namespace gp {
            ///@ingroup model_opt
            ///do not optimize anything
            template <typename Params>
            struct NoLFOpt {
                template <typename GP>
                void operator()(GP&) const
                {
                    std::cerr << "'NoLFOpt' should never be called!" << std::endl;
                    assert(false);
                }
            };
        }
    }
}

#endif
