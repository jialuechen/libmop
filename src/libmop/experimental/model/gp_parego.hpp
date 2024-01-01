
#ifndef libmop_MODEL_GP_PAREGO_HPP
#define libmop_MODEL_GP_PAREGO_HPP

#include <cassert>
#include <iostream>
#include <limits>
#include <vector>

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/LU>

#include <libmop/model/gp/no_lf_opt.hpp>

namespace libmop {
    namespace experimental {
        namespace defaults {
            struct model_gp_parego {
                BO_PARAM(double, rho, 0.05);
            };
        }
        namespace model {

            /// this is the model used in Parego
            /// reference: Knowles, J. (2006). ParEGO: A hybrid algorithm
            /// with on-line landscape approximation for expensive multiobjective
            /// optimization problems.
            /// IEEE Transactions On Evolutionary Computation, 10(1), 50-66.
            /// Main idea:
            /// - this models aggregates all the objective values with the Tchebycheff distance
            /// - objectives are weighted using a random vector
            /// - a single model is built
            template <typename Params, typename Model>
            class GPParego : public Model {
            public:
                GPParego() {}
                GPParego(int dim_in, int dim_out) : Model(dim_in, 1), _nb_objs(dim_out) {}
                void compute(const std::vector<Eigen::VectorXd>& samples,
                    const std::vector<Eigen::VectorXd>& observations)
                {
                    _raw_observations = observations;
                    _nb_objs = observations[0].size();
                    auto new_observations = _scalarize_obs(observations);
                    Model::compute(samples, new_observations);
                }
                /// add sample will NOT be incremental (we call compute each time)
                void add_sample(const Eigen::VectorXd& sample, const Eigen::VectorXd& observation)
                {
                    this->_samples.push_back(sample);
                    _raw_observations.push_back(observation);

                    this->compute(this->_samples, _raw_observations);
                }

            protected:
                size_t _nb_objs;
                std::vector<Eigen::VectorXd> _raw_observations;
                std::vector<Eigen::VectorXd> _scalarize_obs(const std::vector<Eigen::VectorXd>& observations)
                {
                    Eigen::VectorXd lambda = tools::random_vector(_nb_objs);
                    double sum = lambda.sum();
                    lambda = lambda / sum;
                    // scalarize (Tchebycheff)
                    std::vector<Eigen::VectorXd> scalarized;
                    for (auto x : observations) {
                        double y = (lambda.array() * x.array()).maxCoeff();
                        double s = (lambda.array() * x.array()).sum();
                        auto v = tools::make_vector(y + Params::model_gp_parego::rho() * s);
                        scalarized.push_back(v);
                    }
                    return scalarized;
                }
            };
        }
    }
}

#endif
