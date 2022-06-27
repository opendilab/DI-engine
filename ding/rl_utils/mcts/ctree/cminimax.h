#ifndef CMINIMAX_H
#define CMINIMAX_H

#include <iostream>
#include <vector>

const float FLOAT_MAX = 1000000.0;
const float FLOAT_MIN = -FLOAT_MAX;

namespace tools {

    class CMinMaxStats {
        public:
            float maximum, minimum, value_delta_max;

            CMinMaxStats();
            ~CMinMaxStats();

            void set_delta(float value_delta_max);
            void update(float value);
            void clear();
            float normalize(float value);
    };

    class CMinMaxStatsList {
        public:
            int num;
            std::vector<CMinMaxStats> stats_lst;

            CMinMaxStatsList();
            CMinMaxStatsList(int num);
            ~CMinMaxStatsList();

            void set_delta(float value_delta_max);
    };
}

#endif