
#ifndef libmop_OPT_NLOPT_GRAD_HPP
#define libmop_OPT_NLOPT_GRAD_HPP

#ifndef USE_NLOPT
#warning No NLOpt
#else
#include <libmop/opt/nlopt_base.hpp>

namespace libmop {
    namespace defaults {
        struct opt_nloptgrad {
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
         Binding to gradient-based NLOpt algorithms.
         See: http://ab-initio.mit.edu/wiki/index.php/NLopt_Algorithms

         Algorithms:
         - GD_STOGO
         - GD_STOGO_RAND
         - LD_LBFGS_NOCEDAL
         - LD_LBFGS
         - LD_VAR1
         - LD_VAR2
         - LD_TNEWTON
         - LD_TNEWTON_RESTART
         - LD_TNEWTON_PRECOND
         - LD_TNEWTON_PRECOND_RESTART
         - GD_MLSL
         - GD_MLSL_LDS
         - LD_MMA
         - LD_AUGLAG
         - LD_AUGLAG_EQ
         - LD_SLSQP
         - LD_CCSAQ
         - GN_AGS

         Parameters :
         - int iterations
         - double fun_tolerance
         - double xrel_tolerance
        */
        template <typename Params, nlopt::algorithm Algorithm = nlopt::LD_LBFGS>
        struct NLOptGrad : public NLOptBase<Params, Algorithm> {
        public:
            void initialize(int dim) override
            {
                // Assert that the algorithm is gradient-based
                // TO-DO: Add support for MLSL (Multi-Level Single-Linkage)
                // clang-format off
                static_assert(Algorithm == nlopt::LD_MMA || Algorithm == nlopt::LD_SLSQP ||
                    Algorithm == nlopt::LD_LBFGS || Algorithm == nlopt::LD_TNEWTON_PRECOND_RESTART ||
                    Algorithm == nlopt::LD_TNEWTON_PRECOND || Algorithm == nlopt::LD_TNEWTON_RESTART ||
                    Algorithm == nlopt::LD_TNEWTON || Algorithm == nlopt::LD_VAR2 ||
                    Algorithm == nlopt::LD_VAR1 || Algorithm == nlopt::GD_STOGO ||
                    Algorithm == nlopt::GD_STOGO_RAND || Algorithm == nlopt::LD_LBFGS_NOCEDAL ||
                    Algorithm == nlopt::LD_AUGLAG || Algorithm == nlopt::LD_AUGLAG_EQ ||
                    Algorithm == nlopt::LD_CCSAQ, "NLOptGrad accepts gradient-based nlopt algorithms only");
                // clang-format on

                NLOptBase<Params, Algorithm>::initialize(dim);

                this->_opt.set_maxeval(Params::opt_nloptgrad::iterations());
                this->_opt.set_ftol_rel(Params::opt_nloptgrad::fun_tolerance());
                this->_opt.set_xtol_rel(Params::opt_nloptgrad::xrel_tolerance());
            }
        };
    } // namespace opt
} // namespace libmop

#endif
#endif
