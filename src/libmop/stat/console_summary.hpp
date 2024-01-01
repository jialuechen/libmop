
#ifndef libmop_STAT_CONSOLE_SUMMARY_HPP
#define libmop_STAT_CONSOLE_SUMMARY_HPP

#include <libmop/stat/stat_base.hpp>

namespace libmop {
    namespace stat {
        ///@ingroup stat
        ///write the status of the algorithm on the terminal
        template <typename Params>
        struct ConsoleSummary : public StatBase<Params> {
            template <typename BO, typename AggregatorFunction>
            void operator()(const BO& bo, const AggregatorFunction& afun)
            {
                if (!bo.stats_enabled() || bo.observations().empty())
                    return;

                std::cout << bo.total_iterations() << " new point: "
                          << bo.samples().back().transpose()
                          << " value: " << afun(bo.observations().back())
                          << " best:" << afun(bo.best_observation(afun)) << std::endl;
            }
        };
    }
}

#endif
