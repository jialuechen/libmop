
#ifndef libmop_OPT_HPP
#define libmop_OPT_HPP

///@defgroup opt_defaults
///@defgroup opt

#include <libmop/opt/chained.hpp>
#include <libmop/opt/optimizer.hpp>
#ifdef USE_LIBCMAES
#include <libmop/opt/cmaes.hpp>
#else
#warning NO CMA-ES
#endif
#include <libmop/opt/grid_search.hpp>
#ifdef USE_NLOPT
#include <libmop/opt/nlopt_grad.hpp>
#include <libmop/opt/nlopt_no_grad.hpp>
#endif
#include <libmop/opt/adam.hpp>
#include <libmop/opt/gradient_ascent.hpp>
#include <libmop/opt/parallel_repeater.hpp>
#include <libmop/opt/random_point.hpp>
#include <libmop/opt/rprop.hpp>

#endif
