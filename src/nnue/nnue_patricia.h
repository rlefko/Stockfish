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

// Patricia NNUE architecture: 768x2->768->1 with SCReLU activation
// Based on Patricia engine's aggressive evaluation network

#ifndef NNUE_PATRICIA_H_INCLUDED
#define NNUE_PATRICIA_H_INCLUDED

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>

namespace Stockfish {
class Position;  // Forward declaration
}

namespace Stockfish::Eval::NNUE {

// Patricia network architecture constants
constexpr size_t PATRICIA_INPUT_SIZE = 768;
constexpr size_t PATRICIA_LAYER1_SIZE = 768;

constexpr int PATRICIA_SCRELU_MIN = 0;
constexpr int PATRICIA_SCRELU_MAX = 255;
constexpr int PATRICIA_SCALE = 400;
constexpr int PATRICIA_QA = 255;
constexpr int PATRICIA_QB = 64;
constexpr int PATRICIA_QAB = PATRICIA_QA * PATRICIA_QB;

// Patricia network parameters structure (matches Patricia's NNUE_Params)
struct alignas(64) PatriciaNetParams {
    std::array<int16_t, PATRICIA_INPUT_SIZE * PATRICIA_LAYER1_SIZE> feature_weights;
    std::array<int16_t, PATRICIA_LAYER1_SIZE> feature_bias;
    std::array<int16_t, PATRICIA_LAYER1_SIZE * 2> output_weights;
    int16_t output_bias;
};

// Patricia accumulator (perspective-based)
struct alignas(64) PatriciaAccumulator {
    std::array<int16_t, PATRICIA_LAYER1_SIZE> white;
    std::array<int16_t, PATRICIA_LAYER1_SIZE> black;

    void init(const std::array<int16_t, PATRICIA_LAYER1_SIZE>& bias) {
        std::memcpy(white.data(), bias.data(), sizeof(white));
        std::memcpy(black.data(), bias.data(), sizeof(black));
    }
};

// SCReLU activation (Squared Clipped ReLU)
constexpr int32_t screlu(int16_t x) {
    const auto clipped = std::clamp(static_cast<int32_t>(x),
                                     PATRICIA_SCRELU_MIN,
                                     PATRICIA_SCRELU_MAX);
    return clipped * clipped;
}

// Patricia network evaluation
class PatriciaNetwork {
public:
    PatriciaNetwork() = default;

    // Load network from file
    bool load(const std::string& filename);

    // Load network from memory (INCBIN data)
    void load_from_memory(const void* data);

    // Evaluate position using Patricia network
    int32_t evaluate(const PatriciaAccumulator& accumulator, bool perspective_white) const;

    // Initialize accumulator from position
    void init_accumulator(PatriciaAccumulator& acc) const {
        acc.init(params.feature_bias);
    }

    // Update accumulator for feature changes
    void update_accumulator(PatriciaAccumulator& acc,
                           bool is_white_feature,
                           int feature_index,
                           bool add) const;

    const PatriciaNetParams& get_params() const { return params; }

private:
    PatriciaNetParams params;
};

// Patricia phase types for network switching
enum class PatriciaPhase {
    Middlegame = 0,
    Endgame = 1,
    Sacrifice = 2
};

// Patricia multi-network system
class PatriciaNetworks {
public:
    PatriciaNetworks() = default;

    // Load all three Patricia networks
    bool load_networks(const std::string& firefly_path,
                      const std::string& rw3_path,
                      const std::string& allie_path);

    // Load from embedded data
    void load_from_embedded(const void* firefly_data,
                           const void* rw3_data,
                           const void* allie_data);

    // Get network for current phase
    const PatriciaNetwork& get_network(PatriciaPhase phase) const {
        switch (phase) {
            case PatriciaPhase::Middlegame: return firefly;
            case PatriciaPhase::Endgame: return rw3;
            case PatriciaPhase::Sacrifice: return allie;
            default: return firefly;
        }
    }

    // Evaluate with current phase
    int32_t evaluate(const PatriciaAccumulator& acc,
                    PatriciaPhase phase,
                    bool perspective_white) const {
        return get_network(phase).evaluate(acc, perspective_white);
    }

private:
    PatriciaNetwork firefly;  // Middlegame network
    PatriciaNetwork rw3;      // Endgame network
    PatriciaNetwork allie;    // Sacrifice network
};

// Feature indexing for accumulator updates
// Maps Stockfish Piece and Square to Patricia's 768-dim feature indices
// Returns pair of (white_perspective_index, black_perspective_index)
std::pair<size_t, size_t> feature_indices(int stockfish_piece, int square);

// Refresh accumulator from current position (rebuilds from scratch)
void refresh_accumulator(PatriciaAccumulator& acc,
                         const Stockfish::Position& pos,
                         const PatriciaNetwork& network);

// Global Patricia networks accessor
void init_patricia_networks();
PatriciaNetworks& get_patricia_networks();

}  // namespace Stockfish::Eval::NNUE

#endif  // NNUE_PATRICIA_H_INCLUDED
