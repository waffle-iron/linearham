#ifndef LINEARHAM_YAML_
#define LINEARHAM_YAML_

#include <regex>
#include <unordered_map>
#include "../yaml-cpp/include/yaml-cpp/yaml.h"
#include "germline.hpp"

namespace linearham {

bool is_subset_alphabet(std::vector<std::string> vec,
                        std::vector<std::string> alphabet);

std::pair<std::vector<std::string>, Eigen::VectorXd> parse_transitions(
    YAML::Node node);

std::pair<std::vector<std::string>, Eigen::VectorXd> parse_emissions(
    YAML::Node node);

std::unique_ptr<Germline> parse_germline_yaml(std::string yaml_file);

std::map<std::string, int> parse_string_int_map_yaml(std::string s);

typedef std::map<std::string, std::pair<int, int>> boundsbounds_map;
boundsbounds_map parse_boundsbounds_yaml(std::string yaml_str);
}

#endif  // LINEARHAM_YAML_
