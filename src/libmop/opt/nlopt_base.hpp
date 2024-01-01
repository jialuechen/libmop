
#ifndef libmop_OPT_NLOPT_BASE_HPP
#define libmop_OPT_NLOPT_BASE_HPP

#ifndef USE_NLOPT
#warning No NLOpt
#else
#include <Eigen/Core>

#include <vector>

#include <nlopt.hpp>

#include <libmop/opt/optimizer.hpp>
#include <libmop/tools/macros.hpp>

namespace libmop {
    namespace opt {
        /**
          @ingroup opt
        Base class for NLOpt wrappers
        */
        template <typename Params, nlopt::algorithm Algorithm>
        struct NLOptBase {
        public:
            virtual void initialize(int dim)
            {
                _opt = nlopt::opt(Algorithm, dim);
                _initialized = true;
            }

            template <typename F>
            Eigen::VectorXd operator()(const F& f, const Eigen::VectorXd& init, bool bounded)
            {
                int dim = init.size();
                if (!_initialized)
                    initialize(dim);

                _opt.set_max_objective(nlopt_func<F>, (void*)&f);

                std::vector<double> x(dim);
                Eigen::VectorXd::Map(&x[0], dim) = init;

                if (bounded) {
                    _opt.set_lower_bounds(std::vector<double>(dim, 0));
                    _opt.set_upper_bounds(std::vector<double>(dim, 1));
                }

                double max;

                try {
                    _opt.optimize(x, max);
                }
                catch (nlopt::roundoff_limited& e) {
                    // In theory it's ok to ignore this error
                    std::cerr << "[NLOptNoGrad]: " << e.what() << std::endl;
                }
                catch (std::invalid_argument& e) {
                    // In theory it's ok to ignore this error
                    std::cerr << "[NLOptNoGrad]: " << e.what() << std::endl;
                }
                catch (std::runtime_error& e) {
                    // In theory it's ok to ignore this error
                    std::cerr << "[NLOptGrad]: " << e.what() << std::endl;
                }

                return Eigen::VectorXd::Map(x.data(), x.size());
            }

            // Inequality constraints of the form f(x) <= 0
            template <typename F>
            void add_inequality_constraint(const F& f, double tolerance = 1e-8)
            {
                if (_initialized)
                    _opt.add_inequality_constraint(nlopt_func<F>, (void*)&f, tolerance);
                else
                    std::cerr << "[NLOptNoGrad]: Trying to set an inequality constraint without having initialized the optimizer! Nothing will be done!" << std::endl;
            }

            // Equality constraints of the form f(x) = 0
            template <typename F>
            void add_equality_constraint(const F& f, double tolerance = 1e-8)
            {
                if (_initialized)
                    _opt.add_equality_constraint(nlopt_func<F>, (void*)&f, tolerance);
                else
                    std::cerr << "[NLOptNoGrad]: Trying to set an equality constraint without having initialized the optimizer! Nothing will be done!" << std::endl;
            }

        protected:
            nlopt::opt _opt;
            bool _initialized = false;

            template <typename F>
            static double nlopt_func(const std::vector<double>& x, std::vector<double>& grad, void* my_func_data)
            {
                F* f = (F*)(my_func_data);
                Eigen::VectorXd params = Eigen::VectorXd::Map(x.data(), x.size());
                double v;
                if (!grad.empty()) {
                    auto r = eval_grad(*f, params);
                    v = opt::fun(r);
                    Eigen::VectorXd g = opt::grad(r);
                    Eigen::VectorXd::Map(&grad[0], g.size()) = g;
                }
                else {
                    v = eval(*f, params);
                }
                return v;
            }
        };
    } // namespace opt
} // namespace libmop

#endif
#endif
