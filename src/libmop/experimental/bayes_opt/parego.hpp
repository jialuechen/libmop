
#ifndef libmop_BAYES_OPT_PAREGO_HPP
#define libmop_BAYES_OPT_PAREGO_HPP

#include <algorithm>

#include <libmop/bayes_opt/bo_base.hpp>
#include <libmop/bayes_opt/boptimizer.hpp>
#include <libmop/experimental/model/gp_parego.hpp>
#include <libmop/tools/macros.hpp>

namespace libmop {
    namespace experimental {
        namespace bayes_opt {

            BOOST_PARAMETER_TEMPLATE_KEYWORD(parego_modelfun)

            using parego_signature = boost::parameter::parameters<boost::parameter::optional<tag::parego_modelfun>>;

            // clang-format off
            template <class Params,
                      class A1 = boost::parameter::void_,
                      class A2 = boost::parameter::void_,
                      class A3 = boost::parameter::void_,
                      class A4 = boost::parameter::void_,
                      class A5 = boost::parameter::void_>
            // we find the model a wrap it into a GPParego
            // YOU NEED TO PASS A parego_modelfun and not a modelfun !
            class Parego : public libmop::bayes_opt::BOptimizer<
              Params,
              modelfun<
                  typename model::GPParego<Params,
                    typename boost::parameter::binding<
                         typename parego_signature::bind<A1, A2, A3, A4, A5>::type,
                         tag::parego_modelfun,
                         typename libmop::bayes_opt::BoBase<Params, A1, A2, A3, A4, A5>::defaults::model_t
                    > ::type // end binding
                  > // end GPParego
              > // end model_fun
            , A1, A2, A3, A4, A5> {//pass the remaining arguments
              // nothing here !
            };
            // clang-format on
        }
    }
}

#endif
