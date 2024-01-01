
#ifndef libmop_STAT_PARETO_FRONT_HPP
#define libmop_STAT_PARETO_FRONT_HPP

#include <libmop/experimental/tools/pareto.hpp>
#include <libmop/stat/stat_base.hpp>

namespace libmop {
    namespace experimental {
        namespace stat {
            template <typename Params>
            struct ParetoFront : public libmop::stat::StatBase<Params> {
                // point, obj, sigma
                using pareto_point_t = std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd>;
                using pareto_t = std::vector<pareto_point_t>;

                template <typename BO, typename AggregatorFunction>
                void operator()(const BO& bo, const AggregatorFunction&)
                {
                    if (!bo.stats_enabled() || bo.observations().empty())
                        return;
                    std::string fname = bo.res_dir() + "/" + "pareto_front_" + std::to_string(bo.current_iteration()) + ".dat";
                    std::ofstream ofs(fname.c_str());
                    auto pareto = _pareto_data(bo);
                    for (auto x : pareto) {
                        ofs << std::get<0>(x).transpose() << " "
                            << std::get<1>(x).transpose() << std::endl;
                    }
                }

            protected:
                template <typename BO>
                pareto_t _pareto_data(const BO& bo)
                {
                    std::vector<Eigen::VectorXd> v(bo.samples().size());
                    size_t dim = bo.observations().size();
                    std::fill(v.begin(), v.end(), Eigen::VectorXd::Zero(dim));
                    return pareto::pareto_set<1>(
                        _pack_data(bo.samples(), bo.observations(), v));
                }
                pareto_t _pack_data(const std::vector<Eigen::VectorXd>& points,
                    const std::vector<Eigen::VectorXd>& objs,
                    const std::vector<Eigen::VectorXd>& sigma) const
                {
                    assert(points.size() == objs.size());
                    assert(sigma.size() == objs.size());
                    pareto_t p(points.size());
                    tools::par::loop(0, p.size(), [&](size_t k) {
                        // clang-format off
                    p[k] = std::make_tuple(points[k], objs[k], sigma[k]);
                        // clang-format on
                    });
                    return p;
                }
            };
        }
    }
}

#endif
