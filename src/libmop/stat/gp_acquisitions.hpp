
#ifndef libmop_STAT_GP_ACQUISITIONS_HPP
#define libmop_STAT_GP_ACQUISITIONS_HPP

#include <cmath>

#include <libmop/stat/stat_base.hpp>

namespace libmop {
    namespace stat {
        ///@ingroup stat
        ///filename: `gp_acquisitions.dat`
        template <typename Params>
        struct GPAcquisitions : public StatBase<Params> {
            template <typename BO, typename AggregatorFunction>
            void operator()(const BO& bo, const AggregatorFunction& afun)
            {
                if (!bo.stats_enabled())
                    return;

                this->_create_log_file(bo, "gp_acquisitions.dat");

                if (bo.total_iterations() == 0)
                    (*this->_log_file) << "#iteration mu sigma acquisition" << std::endl;

                Eigen::VectorXd mu;
                double sigma, acqui;

                if (!bo.samples().empty()) {
                    std::tie(mu, sigma) = bo.model().query(bo.samples().back());
                    acqui = opt::fun(typename BO::acquisition_function_t(bo.model(), bo.current_iteration())(bo.samples().back(), afun, false));
                }
                else
                    return;

                (*this->_log_file) << bo.total_iterations() << " " << afun(mu) << " " << sigma << " " << acqui << std::endl;
            }
        };
    }
}

#endif
