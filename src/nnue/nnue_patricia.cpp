/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2025 The Stockfish developers (see AUTHORS file)

  Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Stockfish is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "nnue_patricia.h"

#include <fstream>
#include <iostream>

namespace Stockfish::Eval::NNUE {

bool PatriciaNetwork::load(const std::string& filename) {
    std::ifstream stream(filename, std::ios::binary);
    if (!stream.is_open()) {
        std::cerr << "Failed to open Patricia network file: " << filename << std::endl;
        return false;
    }

    // Read network parameters
    stream.read(reinterpret_cast<char*>(&params), sizeof(PatriciaNetParams));

    if (!stream) {
        std::cerr << "Failed to read Patricia network data from: " << filename << std::endl;
        return false;
    }

    return true;
}

void PatriciaNetwork::load_from_memory(const void* data) {
    std::memcpy(&params, data, sizeof(PatriciaNetParams));
}

int32_t PatriciaNetwork::evaluate(const PatriciaAccumulator& accumulator,
                                   bool perspective_white) const {
    // Select accumulator based on perspective
    const auto& us_acc = perspective_white ? accumulator.white : accumulator.black;
    const auto& them_acc = perspective_white ? accumulator.black : accumulator.white;

    // Apply SCReLU activation and compute output
    int32_t sum = 0;

    // Process "us" accumulator
    for (size_t i = 0; i < PATRICIA_LAYER1_SIZE; ++i) {
        int32_t activated = screlu(us_acc[i]);
        sum += activated * params.output_weights[i];
    }

    // Process "them" accumulator
    for (size_t i = 0; i < PATRICIA_LAYER1_SIZE; ++i) {
        int32_t activated = screlu(them_acc[i]);
        sum += activated * params.output_weights[PATRICIA_LAYER1_SIZE + i];
    }

    // Add output bias and scale
    sum += params.output_bias;
    sum /= PATRICIA_QAB;

    return sum * PATRICIA_SCALE / (PATRICIA_QA * PATRICIA_QA);
}

void PatriciaNetwork::update_accumulator(PatriciaAccumulator& acc,
                                         bool is_white_feature,
                                         int feature_index,
                                         bool add) const {
    auto& target_acc = is_white_feature ? acc.white : acc.black;
    const auto& weights = params.feature_weights;
    const int offset = feature_index * PATRICIA_LAYER1_SIZE;

    if (add) {
        for (size_t i = 0; i < PATRICIA_LAYER1_SIZE; ++i) {
            target_acc[i] += weights[offset + i];
        }
    } else {
        for (size_t i = 0; i < PATRICIA_LAYER1_SIZE; ++i) {
            target_acc[i] -= weights[offset + i];
        }
    }
}

bool PatriciaNetworks::load_networks(const std::string& firefly_path,
                                     const std::string& rw3_path,
                                     const std::string& allie_path) {
    bool success = true;

    if (!firefly.load(firefly_path)) {
        std::cerr << "Failed to load Firefly network" << std::endl;
        success = false;
    }

    if (!rw3.load(rw3_path)) {
        std::cerr << "Failed to load RW3 network" << std::endl;
        success = false;
    }

    if (!allie.load(allie_path)) {
        std::cerr << "Failed to load Allie network" << std::endl;
        success = false;
    }

    return success;
}

void PatriciaNetworks::load_from_embedded(const void* firefly_data,
                                          const void* rw3_data,
                                          const void* allie_data) {
    firefly.load_from_memory(firefly_data);
    rw3.load_from_memory(rw3_data);
    allie.load_from_memory(allie_data);
}

}  // namespace Stockfish::Eval::NNUE
