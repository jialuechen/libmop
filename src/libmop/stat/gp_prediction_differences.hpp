
#ifndef libmop_STAT_GP_PREDICTION_DIFFERENCES_HPP
#define libmop_STAT_GP_PREDICTION_DIFFERENCES_HPP

#include <cmath>

#include <libmop/stat/stat_base.hpp>

namespace libmop {
    namespace stat {
        ///@ingroup stat
        ///filename: `gp_prediction_differences.dat`
        template <typename Params>
        struct GPPredictionDifferences : public StatBase<Params> {
            template <typename BO, typename AggregatorFunction>
            void operator()(const BO& bo, const AggregatorFunction& afun)
            {
                if (!bo.stats_enabled() || bo.observations().empty())
                    return;

                this->_create_log_file(bo, "gp_prediction_differences.dat");

                if (bo.total_iterations() == 0)
                    (*this->_log_file) << "#iteration predicted observed difference" << std::endl;

                double pred = afun(bo.model().mu(bo.samples().back()));
                double obs = afun(bo.observations().back());
                (*this->_log_file) << bo.total_iterations() << " " << pred << " " << obs << " " << std::abs(pred - obs) << std::endl;
            }
        };
    }
}

#endif
