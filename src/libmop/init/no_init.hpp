
#ifndef libmop_INIT_NO_INIT_HPP
#define libmop_INIT_NO_INIT_HPP

namespace libmop {
    namespace init {
        ///@ingroup init
        ///Do nothing (dummy initializer).
        template <typename Params>
        struct NoInit {
            template <typename StateFunction, typename AggregatorFunction, typename Opt>
            void operator()(const StateFunction&, const AggregatorFunction&, Opt&) const {}
        };
    }
}

#endif
