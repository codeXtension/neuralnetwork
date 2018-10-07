//
// Created by elie on 03.10.18.
//

#ifndef NEURALNETWORK_LOG_H
#define NEURALNETWORK_LOG_H

#include <type_traits>
#include <iostream>
#include <chrono>
#include <ctime>
#include <iostream>
#include <map>
#include <string>
#include "utilities.h"

using std::chrono::system_clock;
namespace cx {

    class log {
    private:
        std::ostream &_out_stream;
        bool _next_is_begin;
        LogLevel log_level;

        const std::string _log_header;
        using endl_type = std::ostream &(
                std::ostream &); //This is the key: std::endl is a template function, and this is the signature of that function (For std::ostream).

    public:
        static const std::string default_log_header;

        //Constructor: User passes a custom log header and output stream, or uses defaults.
        log(const LogLevel &logLevel = INFO, const std::string &log_header = default_log_header,
            std::ostream &out_stream = std::cout) : log_level(logLevel), _log_header(log_header),
                                                    _out_stream(out_stream),
                                                    _next_is_begin(true) {}

        //Overload for std::endl only:
        log &operator<<(endl_type endl) {
            _next_is_begin = true;

            if (log_level >= DEFAULT_LOG_LEVEL) {
                _out_stream << endl;
            }
            return *this;
        }

        //Overload for anything else:
        template<typename T>
        log &operator<<(const T &data) {
            auto now = std::chrono::system_clock::now();
            auto now_time_t = std::chrono::system_clock::to_time_t(now); //Uhhg, C APIs...
            auto now_tm = std::localtime(&now_time_t); //More uhhg, C style...
            string my_header = "";
            if (_log_header != "") {
                my_header = " [" + _log_header + "] ";
            }
            if (log_level >= DEFAULT_LOG_LEVEL) {
                if (_next_is_begin)
                    _out_stream << "[" << log_level << "][" << now_tm->tm_mday << "." << now_tm->tm_mon << "."
                                << (now_tm->tm_year + 1900) << " " << now_tm->tm_hour << ":" << now_tm->tm_min << ":"
                                << now_tm->tm_sec << "]" << my_header << "\t" << data;
                else
                    _out_stream << data;
            }

            _next_is_begin = false;

            return *this;
        }
    };

    const std::string log::default_log_header = "";
}


#endif //NEURALNETWORK_LOG_H
