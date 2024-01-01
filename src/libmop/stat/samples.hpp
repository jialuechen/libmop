
#ifndef libmop_STAT_SAMPLES_HPP
#define libmop_STAT_SAMPLES_HPP

#include <libmop/stat/stat_base.hpp>

namespace libmop {
    namespace stat {
        ///@ingroup stat
        ///filename: `samples.dat`
        template <typename Params>
        struct Samples : public StatBase<Params> {
            template <typename BO, typename AggregatorFunction>
            void operator()(const BO& bo, const AggregatorFunction&)
            {
                if (!bo.stats_enabled() || bo.samples().empty())
                    return;

                this->_create_log_file(bo, "samples.dat");

                if (bo.total_iterations() == 0) {
                    (*this->_log_file) << "#iteration sample" << std::endl;
                    for (size_t i = 0; i < bo.samples().size() - 1; i++)
                        (*this->_log_file) << "-1 " << bo.samples()[i].transpose() << std::endl;
                }

                (*this->_log_file) << bo.total_iterations() << " " << bo.samples().back().transpose() << std::endl;
            }
        };
    }
}

#endif
