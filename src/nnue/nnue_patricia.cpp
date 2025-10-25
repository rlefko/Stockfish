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

#include "../position.h"
#include "../types.h"

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

    // Patricia's evaluation formula (from Patricia engine nnue.h:295):
    // output = sum / QA
    // result = (output + bias) * SCALE / QAB
    int32_t output = sum / PATRICIA_QA;
    return (output + params.output_bias) * PATRICIA_SCALE / PATRICIA_QAB;
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

// Feature indexing for Patricia NNUE
// Patricia uses piece encoding: 2*piece_type + color (piece_type: 0-5, color: 0=white/1=black)
// Stockfish uses: W_PAWN=1, W_KNIGHT=2, ..., B_PAWN=9, B_KNIGHT=10, ...
std::pair<size_t, size_t> feature_indices(int stockfish_piece, int square) {
    constexpr size_t color_stride = 64 * 6;  // 384 features per color
    constexpr size_t piece_stride = 64;      // 64 squares per piece type

    // Extract piece type and color from Stockfish encoding
    // Stockfish: W_PAWN=1, ..., W_KING=6, B_PAWN=9, ..., B_KING=14
    const int piece_type = (stockfish_piece - 1) % 8;  // 0=PAWN, 1=KNIGHT, ..., 5=KING
    const int color = (stockfish_piece >= 9) ? 1 : 0;  // 0=white, 1=black

    // Patricia's feature layout:
    // White perspective: [white pieces 0-383][black pieces 384-767]
    // Black perspective: same but with squares flipped (sq ^ 56)
    const auto white_idx = color * color_stride + piece_type * piece_stride + static_cast<size_t>(square);
    const auto black_idx = (color ^ 1) * color_stride + piece_type * piece_stride + (static_cast<size_t>(square ^ 56));

    return {white_idx, black_idx};
}

// Refresh accumulator by rebuilding from current position
void refresh_accumulator(PatriciaAccumulator& acc,
                         const Stockfish::Position& pos,
                         const PatriciaNetwork& network) {
    // Start with bias
    network.init_accumulator(acc);

    // Add features for all pieces on the board
    for (int sq = 0; sq < 64; ++sq) {
        int piece = pos.piece_on(static_cast<Stockfish::Square>(sq));
        if (piece != Stockfish::NO_PIECE) {
            const auto [white_idx, black_idx] = feature_indices(piece, sq);

            // Add to white perspective accumulator
            const auto& weights = network.get_params().feature_weights;
            const int white_offset = white_idx * PATRICIA_LAYER1_SIZE;
            for (size_t i = 0; i < PATRICIA_LAYER1_SIZE; ++i) {
                acc.white[i] += weights[white_offset + i];
            }

            // Add to black perspective accumulator
            const int black_offset = black_idx * PATRICIA_LAYER1_SIZE;
            for (size_t i = 0; i < PATRICIA_LAYER1_SIZE; ++i) {
                acc.black[i] += weights[black_offset + i];
            }
        }
    }
}

}  // namespace Stockfish::Eval::NNUE
