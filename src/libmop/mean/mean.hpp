#ifndef libmop_MEAN_MEAN_HPP
#define libmop_MEAN_MEAN_HPP

#include <Eigen/Core>

#include <libmop/tools/macros.hpp>

namespace libmop {
    namespace mean {
        /** @ingroup mean
          \rst
          Base struct for mean definition.
          \endrst
        */
        template <typename Params>
        struct BaseMean {
            BaseMean(size_t dim_out = 1) {}

            size_t h_params_size() const { return 0; }

            Eigen::VectorXd h_params() const { return Eigen::VectorXd(); }

            void set_h_params(const Eigen::VectorXd& p) {}

            template <typename GP>
            Eigen::MatrixXd grad(const Eigen::VectorXd& x, const GP& gp) const
            {
                // This should never be called!
                assert(false);
                return Eigen::VectorXd();
            }
        };
    } // namespace mean
} // namespace libmop

#endif
