// Copyright (c) 2023-2026 Takehiro Yoshioka
// SPDX-License-Identifier: MIT

#ifndef READCSV_HPP
#define READCSV_HPP

#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <stdexcept>
#include <string>

// Function to load data from a file into a 2D vector. Blank lines are skipped.
// Throws std::runtime_error if the file cannot be opened.
std::vector<std::vector<double>> readcsv(const std::string& filename, char delimiter = ',') {
    std::ifstream file(filename);
    std::vector<std::vector<double>> data;

    if (!file.is_open()) {
        throw std::runtime_error("cannot open file: " + filename);
    }

    std::string line;
    while (std::getline(file, line)) {
        if (line.find_first_not_of(" \t\r") == std::string::npos) continue;

        std::vector<double> row;
        std::stringstream ss(line);
        std::string token;

        while (std::getline(ss, token, delimiter)) {
            row.push_back(std::stod(token));
        }

        data.push_back(row);
    }

    return data;
}

// Throws std::runtime_error unless `table` has at least `min_rows` rows and
// every row has at least `min_cols` columns.
inline void requireTableShape(std::vector<std::vector<double>> const& table,
                              std::size_t min_rows, std::size_t min_cols,
                              std::string const& name) {
    bool ok = table.size() >= min_rows;
    for (auto const& row : table) ok = ok && row.size() >= min_cols;
    if (!ok) {
        throw std::runtime_error(name + " must have at least " + std::to_string(min_rows) + " rows of "
                                 + std::to_string(min_cols) + " columns");
    }
}

#endif // READCSV_HPP