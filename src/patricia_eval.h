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

// Patricia aggressive evaluation system
// Implements Patricia's 6 aggressiveness features on top of Stockfish search

#ifndef PATRICIA_EVAL_H_INCLUDED
#define PATRICIA_EVAL_H_INCLUDED

#include "nnue/nnue_patricia.h"
#include "types.h"

namespace Stockfish {

class Position;

namespace Patricia {

// Patricia evaluation state
struct PatriciaState {
    Eval::NNUE::PatriciaPhase phase = Eval::NNUE::PatriciaPhase::Middlegame;
    Eval::NNUE::PatriciaAccumulator accumulator;
    int last_phase_check_depth = 0;

    // Material tracking for sacrifice detection
    int starting_material = 0;
    int material_diff_history[256] = {0};
    int history_length = 0;
};

// Initialize Patricia state for a position
void init_patricia_state(PatriciaState& state, const Position& pos);

// Detect current phase based on evaluation
Eval::NNUE::PatriciaPhase detect_phase(Value eval, int depth);

// Evaluate using Patricia's aggressive NNUE + modifiers
Value evaluate_patricia(const Position& pos,
                        PatriciaState& state,
                        int depth,
                        int search_ply);

// Patricia's 6 aggressiveness modifiers
namespace Modifiers {

// 1. Draw contempt: ±50 based on material advantage
Value draw_contempt(const Position& pos, Value base_score);

// 2. Asymmetric sacrifice bonuses (only for original side to move)
Value sacrifice_bonus(const Position& pos,
                     const PatriciaState& state,
                     Value eval,
                     int search_ply);

// 3. Material scaling when ahead
Value material_scaling(Value eval, const Position& pos);

// 4. Better than material bonus
Value better_than_material(Value eval, const Position& pos);

}  // namespace Modifiers

}  // namespace Patricia

}  // namespace Stockfish

#endif  // PATRICIA_EVAL_H_INCLUDED
