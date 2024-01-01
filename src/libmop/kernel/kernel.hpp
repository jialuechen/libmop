#ifndef libmop_KERNEL_KERNEL_HPP
#define libmop_KERNEL_KERNEL_HPP

#include <Eigen/Core>

#include <libmop/tools/macros.hpp>

namespace libmop {
    namespace defaults {
        struct kernel {
            /// @ingroup kernel_defaults
            BO_PARAM(double, noise, 0.01);
            BO_PARAM(bool, optimize_noise, false);
        };
    } // namespace defaults

    namespace kernel {
        /**
          @ingroup kernel
          \rst
          Base struct for kernel definition. It handles the noise and its optimization (only if the kernel allows hyper-parameters optimization).
          \endrst

          Parameters:
             - ``double noise`` (initial signal noise squared)
             - ``bool optimize_noise`` (whether we are optimizing for the noise or not)
        */
        template <typename Params, typename Kernel>
        struct BaseKernel {
        public:
            BaseKernel(size_t dim = 1) : _noise(Params::kernel::noise())
            {
                _noise_p = std::log(std::sqrt(_noise));
            }

            double operator()(const Eigen::VectorXd& v1, const Eigen::VectorXd& v2, int i = -1, int j = -2) const
            {
                return static_cast<const Kernel*>(this)->kernel(v1, v2) + ((i == j) ? _noise + 1e-8 : 0.0);
            }

            Eigen::VectorXd grad(const Eigen::VectorXd& x1, const Eigen::VectorXd& x2, int i = -1, int j = -2) const
            {
                Eigen::VectorXd g = static_cast<const Kernel*>(this)->gradient(x1, x2);

                if (Params::kernel::optimize_noise()) {
                    g.conservativeResize(g.size() + 1);
                    g(g.size() - 1) = ((i == j) ? 2.0 * _noise : 0.0);
                }

                return g;
            }

            // Get the hyper parameters size
            size_t h_params_size() const
            {
                return static_cast<const Kernel*>(this)->params_size() + (Params::kernel::optimize_noise() ? 1 : 0);
            }

            // Get the hyper parameters in log-space
            Eigen::VectorXd h_params() const
            {
                Eigen::VectorXd params = static_cast<const Kernel*>(this)->params();
                if (Params::kernel::optimize_noise()) {
                    params.conservativeResize(params.size() + 1);
                    params(params.size() - 1) = _noise_p;
                }
                return params;
            }

            // We expect the input parameters to be in log-space
            void set_h_params(const Eigen::VectorXd& p)
            {
                static_cast<Kernel*>(this)->set_params(p.head(h_params_size() - (Params::kernel::optimize_noise() ? 1 : 0)));
                if (Params::kernel::optimize_noise()) {
                    _noise_p = p(h_params_size() - 1);
                    _noise = std::exp(2 * _noise_p);
                }
            }

            // Get signal noise
            double noise() const { return _noise; }

        protected:
            double _noise;
            double _noise_p;

            // Functions for compilation issues
            // They should never be called like this
            size_t params_size() const { return 0; }

            Eigen::VectorXd params() const { return Eigen::VectorXd(); }

            void set_params(const Eigen::VectorXd& p) {}

            Eigen::VectorXd gradient(const Eigen::VectorXd& x1, const Eigen::VectorXd& x2) const
            {
                // This should never be called!
                assert(false);
                return Eigen::VectorXd();
            }
        };
    } // namespace kernel
} // namespace libmop

#endif
