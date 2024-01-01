
#ifndef libmop_STAT_GP_KERNEL_HPARAMS_HPP
#define libmop_STAT_GP_KERNEL_HPARAMS_HPP

#include <cmath>

#include <libmop/stat/stat_base.hpp>

namespace libmop {
    namespace stat {
        ///@ingroup stat
        ///filename: `gp_kernel_hparams.dat`
        template <typename Params>
        struct GPKernelHParams : public StatBase<Params> {
            template <typename BO, typename AggregatorFunction>
            void operator()(const BO& bo, const AggregatorFunction& afun)
            {
                if (!bo.stats_enabled() || bo.observations().empty())
                    return;

                this->_create_log_file(bo, "gp_kernel_hparams.dat");

                if (bo.total_iterations() == 0)
                    (*this->_log_file) << "#iteration gp_kernel_hparams" << std::endl;

                (*this->_log_file) << bo.total_iterations() << " " << bo.model().kernel_function().h_params().transpose() << std::endl;
            }
        };
    }
}

#endif
