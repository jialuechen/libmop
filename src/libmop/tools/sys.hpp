

#ifndef libmop_TOOLS_SYS_HPP
#define libmop_TOOLS_SYS_HPP

#include <ctime>
#include <string>

#if defined(__MINGW32__) || defined(_MSC_VER)
#include <winsock2.h>
#else
#include <unistd.h>
#endif

namespace libmop {
    namespace tools {
        /// @ingroup tools
        /// easy way to get the current date
        inline std::string date()
        {
            char date[30];
            time_t date_time;
            time(&date_time);
            strftime(date, 30, "%Y-%m-%d_%H_%M_%S", localtime(&date_time));
            return std::string(date);
        }

        /// @ingroup tools
        /// easy way to get the hostame
        inline std::string hostname()
        {
            char hostname[50];
            int res = gethostname(hostname, 50);
            assert(res == 0);
            res = 0; // avoid a warning in opt mode
            return std::string(hostname);
        }

        /// @ingroup tools
        /// easy way to get the PID
        inline std::string getpid()
        {
            return std::to_string(::getpid());
        }
    }
}

#endif
