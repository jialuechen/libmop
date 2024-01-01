
#ifndef libmop_BAYES_OPT_CBOPTIMIZER_HPP
#define libmop_BAYES_OPT_CBOPTIMIZER_HPP

#include <algorithm>
#include <iostream>
#include <iterator>

#include <boost/parameter/aux_/void.hpp>

#include <Eigen/Core>

#include <libmop/bayes_opt/bo_base.hpp>
#include <libmop/tools/macros.hpp>
#include <libmop/tools/random_generator.hpp>
#ifdef USE_NLOPT
#include <libmop/opt/nlopt_no_grad.hpp>
#elif defined USE_LIBCMAES
#include <libmop/opt/cmaes.hpp>
#else
#include <libmop/opt/grid_search.hpp>
#endif

namespace libmop {
    namespace defaults {
        struct bayes_opt_cboptimizer {
            BO_PARAM(int, hp_period, -1);
            BO_PARAM(bool, bounded, true);
        };
    }

    namespace experimental {
        BOOST_PARAMETER_TEMPLATE_KEYWORD(constraint_modelfun)

        namespace bayes_opt {

            using cboptimizer_signature = boost::parameter::parameters<boost::parameter::optional<libmop::tag::acquiopt>,
                boost::parameter::optional<libmop::tag::statsfun>,
                boost::parameter::optional<libmop::tag::initfun>,
                boost::parameter::optional<libmop::tag::acquifun>,
                boost::parameter::optional<libmop::tag::stopcrit>,
                boost::parameter::optional<libmop::tag::modelfun>,
                boost::parameter::optional<libmop::experimental::tag::constraint_modelfun>>;

            // clang-format off
        /**
        The classic Bayesian optimization algorithm.

        \\rst
        References: :cite:`brochu2010tutorial,Mockus2013`
        \\endrst

        This class takes the same template parameters as BoBase. It adds:
        \\rst
        +---------------------+------------+----------+---------------+
        |type                 |typedef     | argument | default       |
        +=====================+============+==========+===============+
        |acqui. optimizer     |acqui_opt_t | acquiopt | see below     |
        +---------------------+------------+----------+---------------+
        \\endrst

        The default value of acqui_opt_t is:
        - ``opt::Cmaes<Params>`` if libcmaes was found in `waf configure`
        - ``opt::NLOptNoGrad<Params, nlopt::GN_DIRECT_L_RAND>`` if NLOpt was found but libcmaes was not found
        - ``opt::GridSearch<Params>`` otherwise (please do not use this: the algorithm will not work at all!)
        */
        template <class Params,
          class A1 = boost::parameter::void_,
          class A2 = boost::parameter::void_,
          class A3 = boost::parameter::void_,
          class A4 = boost::parameter::void_,
          class A5 = boost::parameter::void_,
          class A6 = boost::parameter::void_,
          class A7 = boost::parameter::void_>
            // clang-format on
            class CBOptimizer : public libmop::bayes_opt::BoBase<Params, A1, A2, A3, A4, A5, A6> {
            public:
                // defaults
                struct defaults {
#ifdef USE_NLOPT
                    using acquiopt_t = libmop::opt::NLOptNoGrad<Params, nlopt::GN_DIRECT_L_RAND>;
#elif defined(USE_LIBCMAES)
                    using acquiopt_t = libmop::opt::Cmaes<Params>;
#else
#warning NO NLOpt, and NO Libcmaes: the acquisition function will be optimized by a grid search algorithm (which is usually bad). Please install at least NLOpt or libcmaes to use libmop!.
                    using acquiopt_t = libmop::opt::GridSearch<Params>;
#endif
                    using kf_t = libmop::kernel::Exp<Params>;
                    using mean_t = libmop::mean::Constant<Params>;
                    using constraint_model_t = libmop::model::GP<Params, kf_t, mean_t>;
                };
                /// link to the corresponding BoBase (useful for typedefs)
                using base_t = libmop::bayes_opt::BoBase<Params, A1, A2, A3, A4, A5, A6>;
                using model_t = typename base_t::model_t;

                using acquisition_function_t = typename base_t::acquisition_function_t;
                // extract the types
                using args = typename cboptimizer_signature::bind<A1, A2, A3, A4, A5, A6, A7>::type;
                using acqui_optimizer_t = typename boost::parameter::binding<args, libmop::tag::acquiopt, typename defaults::acquiopt_t>::type;

                using constraint_model_t = typename boost::parameter::binding<args, libmop::experimental::tag::constraint_modelfun, typename defaults::constraint_model_t>::type;

                /// The main function (run the Bayesian optimization algorithm)
                template <typename StateFunction, typename AggregatorFunction = FirstElem>
                void optimize(const StateFunction& sfun, const AggregatorFunction& afun = AggregatorFunction(), bool reset = true)
                {
                    _nb_constraints = StateFunction::nb_constraints();
                    _dim_out = StateFunction::dim_out();

                    this->_init(sfun, afun, reset);

                    if (!this->_observations.empty()) {
                        _split_observations();
                        _model.compute(this->_samples, _obs[0]);
                        if (_nb_constraints > 0)
                            _constraint_model.compute(this->_samples, _obs[1]);
                    }
                    else {
                        _model = model_t(StateFunction::dim_in(), StateFunction::dim_out());
                        if (_nb_constraints > 0)
                            _constraint_model = constraint_model_t(StateFunction::dim_in(), _nb_constraints);
                    }

                    acqui_optimizer_t acqui_optimizer;

                    while (!this->_stop(*this, afun)) {
                        acquisition_function_t acqui(_model, _constraint_model, this->_current_iteration);

                        auto acqui_optimization =
                            [&](const Eigen::VectorXd& x, bool g) { return acqui(x, afun, g); };
                        Eigen::VectorXd starting_point = tools::random_vector(StateFunction::dim_in(), Params::bayes_opt_cboptimizer::bounded());
                        Eigen::VectorXd new_sample = acqui_optimizer(acqui_optimization, starting_point, Params::bayes_opt_cboptimizer::bounded());
                        this->eval_and_add(sfun, new_sample);

                        this->_update_stats(*this, afun);

                        _model.add_sample(this->_samples.back(), _obs[0].back());
                        if (_nb_constraints > 0)
                            _constraint_model.add_sample(this->_samples.back(), _obs[1].back());

                        if (Params::bayes_opt_cboptimizer::hp_period() > 0
                            && (this->_current_iteration + 1) % Params::bayes_opt_cboptimizer::hp_period() == 0) {
                            _model.optimize_hyperparams();
                            if (_nb_constraints > 0)
                                _constraint_model.optimize_hyperparams();
                        }

                        this->_current_iteration++;
                        this->_total_iterations++;
                    }
                }

                /// return the best observation so far (i.e. max(f(x)))
                template <typename AggregatorFunction = FirstElem>
                const Eigen::VectorXd& best_observation(const AggregatorFunction& afun = AggregatorFunction()) const
                {
                    _dim_out = _model.dim_out();
                    _split_observations();

                    std::vector<Eigen::VectorXd> obs = _feasible_observations();
                    if (obs.size() > 0)
                        _obs[0] = obs;

                    auto rewards = std::vector<double>(_obs[0].size());
                    std::transform(_obs[0].begin(), _obs[0].end(), rewards.begin(), afun);
                    auto max_e = std::max_element(rewards.begin(), rewards.end());
                    return _obs[0][std::distance(rewards.begin(), max_e)];
                }

                /// return the best sample so far (i.e. the argmax(f(x)))
                template <typename AggregatorFunction = FirstElem>
                const Eigen::VectorXd& best_sample(const AggregatorFunction& afun = AggregatorFunction()) const
                {
                    _dim_out = _model.dim_out();
                    _split_observations();

                    std::vector<Eigen::VectorXd> obs = _feasible_observations();
                    if (obs.size() > 0)
                        _obs[0] = obs;

                    auto rewards = std::vector<double>(_obs[0].size());
                    std::transform(_obs[0].begin(), _obs[0].end(), rewards.begin(), afun);
                    auto max_e = std::max_element(rewards.begin(), rewards.end());
                    return this->_samples[std::distance(rewards.begin(), max_e)];
                }

                const model_t& model() const { return _model; } // returns model for the objective function
            protected:
                model_t _model;
                constraint_model_t _constraint_model;

                size_t _nb_constraints;
                mutable size_t _dim_out;
                mutable std::vector<std::vector<Eigen::VectorXd>> _obs;

                std::vector<Eigen::VectorXd> _feasible_observations() const
                {
                    std::vector<Eigen::VectorXd> feasible_obs;
                    for (size_t i = 0; i < _obs[0].size(); ++i) {
                        if (_obs[1][i].prod() > 0)
                            feasible_obs.push_back(_obs[0][i]);
                    }

                    return feasible_obs;
                }

                void _split_observations() const
                {
                    _obs.clear();
                    _obs.resize(2);

                    for (size_t i = 0; i < this->_observations.size(); ++i) {
                        assert(size_t(this->_observations[i].size()) == _dim_out + _nb_constraints);

                        Eigen::VectorXd vec_obj(_dim_out);
                        for (size_t j = 0; j < _dim_out; ++j)
                            vec_obj(j) = this->_observations[i](j);
                        _obs[0].push_back(vec_obj);

                        if (_nb_constraints > 0) {
                            Eigen::VectorXd vec_con(_nb_constraints);
                            for (int j = _dim_out, ind = 0; j < this->_observations[i].size(); ++j, ++ind)
                                vec_con(ind) = this->_observations[i](j);
                            _obs[1].push_back(vec_con);
                        }
                    }
                }
            };
        }
    }
}

#endif
