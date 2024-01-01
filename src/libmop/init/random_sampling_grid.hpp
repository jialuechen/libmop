
#ifndef libmop_INIT_RANDOM_SAMPLING_GRID_HPP
#define libmop_INIT_RANDOM_SAMPLING_GRID_HPP

#include <Eigen/Core>

#include <libmop/tools/macros.hpp>
#include <libmop/tools/random_generator.hpp>

namespace libmop {
    namespace defaults {
        struct init_randomsamplinggrid {
            ///@ingroup init_defaults
            BO_PARAM(int, samples, 10);
            ///@ingroup init_defaults
            BO_PARAM(int, bins, 5);
        };
    }
    namespace init {
        /** @ingroup init
          \rst
          Grid-based random sampling: in effect, define a grid and takes random samples from this grid. The number of bins in the grid is ``bins``^number_of_dimensions

          For instance, if bins = 5 and there are 3 inputs dimensions, then the grid is 5*5*5=125, and random points will all be points of this grid.

          Parameters:
            - ``int samples`` (total number of samples)
            - ``int bins`` (number of bins for each dimensions)
          \endrst
        */
        template <typename Params>
        struct RandomSamplingGrid {
            template <typename StateFunction, typename AggregatorFunction, typename Opt>
            void operator()(const StateFunction& seval, const AggregatorFunction&, Opt& opt) const
            {
                // Only works with bounded BO
                assert(Params::bayes_opt_bobase::bounded());

                tools::rgen_int_t rgen(0, Params::init_randomsamplinggrid::bins());
                for (int i = 0; i < Params::init_randomsamplinggrid::samples(); i++) {
                    Eigen::VectorXd new_sample(StateFunction::dim_in());
                    for (size_t i = 0; i < StateFunction::dim_in(); i++)
                        new_sample[i] = rgen.rand() / double(Params::init_randomsamplinggrid::bins());
                    opt.eval_and_add(seval, new_sample);
                }
            }
        };
    }
}

#endif
