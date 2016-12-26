#pragma once

#include "vector.hpp"

#include <boost/tokenizer.hpp>
#include <Eigen/Dense>

#include <vector>
#include <fstream>
#include <string>

namespace VRSGD {

using Eigen::VectorXd;

template <typename VectorDataT, typename LabelT>
void read_libsvm(std::vector<LabeledPoint<VectorDataT, LabelT>>& data_points, std::string filename, int feature_num) {
    std::fstream fs(filename, std::fstream::in);

    while (!fs.eof()) {
        std::string line;
        std::getline(fs, line);
        if (line == "") {
            continue;
        }

        LabeledPoint<VectorDataT, LabelT> data_point(VectorDataT(feature_num), 0);

        boost::char_separator<char> sep(" \t");
        boost::tokenizer<boost::char_separator<char>> tok(line, sep);
        
        bool first_flag = true;
        for (auto& w : tok) {
            if (first_flag) {
                data_point.y = std::stod(w);
                first_flag = false;
            } else {
                boost::char_separator<char> sep2(":");
                boost::tokenizer<boost::char_separator<char>> tok2(w, sep2);
                auto it = tok2.begin();
                int fea = std::stoi(*it) - 1;
                it++;
                double val = std::stod(*it);

                data_point.x.coeffRef(fea, 0) = val;
            }
        }

        data_points.push_back(std::move(data_point));
    }
}

}

