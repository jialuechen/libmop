
#ifndef libmop_OPT_NLOPT_NO_GRAD_HPP
#define libmop_OPT_NLOPT_NO_GRAD_HPP

#ifndef USE_NLOPT
#warning No NLOpt
#else
#include <libmop/opt/nlopt_base.hpp>

namespace libmop {
    namespace defaults {
        struct opt_nloptnograd {
            /// @ingroup opt_defaults
            /// number of calls to the optimized function
            BO_PARAM(int, iterations, 500);
            /// @ingroup opt_defaults
            /// tolerance for convergence: stop when an optimization step (or an
            /// estimate of the optimum) changes the objective function value by
            /// less than tol multiplied by the absolute value of the function
            /// value.
            /// IGNORED if negative
            BO_PARAM(double, fun_tolerance, -1);
            /// @ingroup opt_defaults
            /// tolerance for convergence: stop when an optimization step (or an
            /// estimate of the optimum) changes all the parameter values by
            /// less than tol multiplied by the absolute value of the parameter
            /// value.
            /// IGNORED if negative
            BO_PARAM(double, xrel_tolerance, -1);
        };
    } // namespace defaults
    namespace opt {
        /**
          @ingroup opt
        Binding to gradient-free NLOpt algorithms.
         See: http://ab-initio.mit.edu/wiki/index.php/NLopt_Algorithms

         Algorithms:
         - GN_DIRECT
         - GN_DIRECT_L, [default]
         - GN_DIRECT_L_RAND
         - GN_DIRECT_NOSCAL
         - GN_DIRECT_L_NOSCAL
         - GN_DIRECT_L_RAND_NOSCAL
         - GN_ORIG_DIRECT
         - GN_ORIG_DIRECT_L
         - GN_CRS2_LM
         - GN_MLSL
         - GN_MLSL_LDS
         - GN_ISRES
         - LN_COBYLA
         - LN_AUGLAG_EQ
         - LN_BOBYQA
         - LN_NEWUOA
         - LN_NEWUOA_BOUND
         - LN_PRAXIS
         - LN_NELDERMEAD
         - LN_SBPLX
         - LN_AUGLAG

         Parameters:
         - int iterations
         - double fun_tolerance
         - double xrel_tolerance
        */
        template <typename Params, nlopt::algorithm Algorithm = nlopt::GN_DIRECT_L_RAND>
        struct NLOptNoGrad : public NLOptBase<Params, Algorithm> {
        public:
            void initialize(int dim) override
            {
                // Assert that the algorithm is non-gradient
                // TO-DO: Add support for MLSL (Multi-Level Single-Linkage)
                // TO-DO: Add better support for ISRES (Improved Stochastic Ranking Evolution Strategy)
                // clang-format off
                static_assert(Algorithm == nlopt::LN_COBYLA || Algorithm == nlopt::LN_BOBYQA ||
                    Algorithm == nlopt::LN_NEWUOA || Algorithm == nlopt::LN_NEWUOA_BOUND ||
                    Algorithm == nlopt::LN_PRAXIS || Algorithm == nlopt::LN_NELDERMEAD ||
                    Algorithm == nlopt::LN_SBPLX || Algorithm == nlopt::GN_DIRECT ||
                    Algorithm == nlopt::GN_DIRECT_L || Algorithm == nlopt::GN_DIRECT_L_RAND ||
                    Algorithm == nlopt::GN_DIRECT_NOSCAL || Algorithm == nlopt::GN_DIRECT_L_NOSCAL ||
                    Algorithm == nlopt::GN_DIRECT_L_RAND_NOSCAL || Algorithm == nlopt::GN_ORIG_DIRECT ||
                    Algorithm == nlopt::GN_ORIG_DIRECT_L || Algorithm == nlopt::GN_CRS2_LM ||
                    Algorithm == nlopt::LN_AUGLAG || Algorithm == nlopt::LN_AUGLAG_EQ ||
                    Algorithm == nlopt::GN_ISRES || Algorithm == nlopt::GN_ESCH, "NLOptNoGrad accepts gradient free nlopt algorithms only");
                // clang-format on

                NLOptBase<Params, Algorithm>::initialize(dim);

                this->_opt.set_maxeval(Params::opt_nloptnograd::iterations());
                this->_opt.set_ftol_rel(Params::opt_nloptnograd::fun_tolerance());
                this->_opt.set_xtol_rel(Params::opt_nloptnograd::xrel_tolerance());
            }
        };
    } // namespace opt
} // namespace libmop

#endif
#endif
