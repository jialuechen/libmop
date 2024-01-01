
#ifndef libmop_STAT_BEST_AGGREGATED_OBSERVATIONS_HPP
#define libmop_STAT_BEST_AGGREGATED_OBSERVATIONS_HPP

#include <libmop/stat/stat_base.hpp>

namespace libmop {
    namespace stat {
        ///@ingroup stat
        /// TODO fallocati
        /// Write the best observation at each iteration
        /// filename: `best_aggregated_observations.dat`
        template <typename Params>
        struct BestAggregatedObservations : public StatBase<Params> {
            template <typename BO, typename AggregatorFunction>
            void operator()(const BO& bo, const AggregatorFunction& afun)
            {
                if (!bo.stats_enabled() || bo.observations().empty())
                    return;

                this->_create_log_file(bo, "best_aggregated_observations.dat");

                if (bo.total_iterations() == 0)
                    (*this->_log_file) << "#iteration best_aggregated_observation" << std::endl;

                (*this->_log_file) << bo.total_iterations() << " " << afun(bo.best_observation(afun)) << std::endl;
            }
        };
    }
}

#endif
