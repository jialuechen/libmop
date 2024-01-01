
#ifndef libmop_STAT_OBSERVATIONS_HPP
#define libmop_STAT_OBSERVATIONS_HPP

#include <libmop/stat/stat_base.hpp>

namespace libmop {
    namespace stat {
        ///@ingroup stat
        ///filename: `observations.dat`
        template <typename Params>
        struct Observations : public StatBase<Params> {
            template <typename BO, typename AggregatorFunction>
            void operator()(const BO& bo, const AggregatorFunction&)
            {
                if (!bo.stats_enabled() || bo.observations().empty())
                    return;

                this->_create_log_file(bo, "observations.dat");

                if (bo.total_iterations() == 0) {
                    (*this->_log_file) << "#iteration observation" << std::endl;
                    for (size_t i = 0; i < bo.observations().size() - 1; i++)
                        (*this->_log_file) << "-1 " << bo.observations()[i].transpose() << std::endl;
                }

                (*this->_log_file) << bo.total_iterations() << " " << bo.observations().back().transpose() << std::endl;
            }
        };
    }
}

#endif
