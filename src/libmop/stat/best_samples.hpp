
#ifndef libmop_STAT_BEST_SAMPLES_HPP
#define libmop_STAT_BEST_SAMPLES_HPP

#include <libmop/stat/stat_base.hpp>

namespace libmop {
    namespace stat {
        ///@ingroup stat
        ///filename: `best_samples.dat`
        template <typename Params>
        struct BestSamples : public StatBase<Params> {
            template <typename BO, typename AggregatorFunction>
            void operator()(const BO& bo, const AggregatorFunction& afun)
            {
                if (!bo.stats_enabled() || bo.samples().empty())
                    return;

                this->_create_log_file(bo, "best_samples.dat");

                if (bo.total_iterations() == 0)
                    (*this->_log_file) << "#iteration best_sample" << std::endl;

                (*this->_log_file) << bo.total_iterations() << " " << bo.best_sample(afun).transpose() << std::endl;
            }
        };
    }
}

#endif
