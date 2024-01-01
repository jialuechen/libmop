
#ifndef libmop_STAT_AGGREGATED_OBSERVATIONS_HPP
#define libmop_STAT_AGGREGATED_OBSERVATIONS_HPP

#include <libmop/stat/stat_base.hpp>

namespace libmop {
    namespace stat {
        ///@ingroup stat
        /// TODO fallocati
        ///Write all the observations
        ///filename: `aggregated_observations.dat`
        template <typename Params>
        struct AggregatedObservations : public StatBase<Params> {
            template <typename BO, typename AggregatorFunction>
            void operator()(const BO& bo, const AggregatorFunction& afun)
            {
                if (!bo.stats_enabled() || bo.observations().empty())
                    return;

                this->_create_log_file(bo, "aggregated_observations.dat");

                if (bo.total_iterations() == 0) {
                    (*this->_log_file) << "#iteration aggregated_observation" << std::endl;
                    for (size_t i = 0; i < bo.observations().size() - 1; i++)
                        (*this->_log_file) << "-1 " << afun(bo.observations()[i]) << std::endl;
                }

                (*this->_log_file) << bo.total_iterations() << " " << afun(bo.observations().back()) << std::endl;
            }
        };
    }
}

#endif
