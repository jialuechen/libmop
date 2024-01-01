
#ifndef libmop_MEAN_NULL_FUNCTION_HPP
#define libmop_MEAN_NULL_FUNCTION_HPP

#include <libmop/mean/mean.hpp>

namespace libmop {
    namespace mean {
        /// @ingroup mean
        /// Constant with m=0
        template <typename Params>
        struct NullFunction : public BaseMean<Params> {
            NullFunction(size_t dim_out = 1) : _dim_out(dim_out) {}

            template <typename GP>
            Eigen::VectorXd operator()(const Eigen::VectorXd& v, const GP&) const
            {
                return Eigen::VectorXd::Zero(_dim_out);
            }

        protected:
            size_t _dim_out;
        };
    }
}

#endif
