
#ifndef libmop_MEAN_DATA_HPP
#define libmop_MEAN_DATA_HPP

#include <libmop/mean/mean.hpp>

namespace libmop {
    namespace mean {
        ///@ingroup mean
        ///Use the mean of the observation as a constant mean
        template <typename Params>
        struct Data : public BaseMean<Params> {
            Data(size_t dim_out = 1) {}

            template <typename GP>
            Eigen::VectorXd operator()(const Eigen::VectorXd& v, const GP& gp) const
            {
                return gp.mean_observation().array();
            }
        };
    } // namespace mean
} // namespace libmop

#endif
