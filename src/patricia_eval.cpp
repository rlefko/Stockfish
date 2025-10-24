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

#include "patricia_eval.h"

#include "position.h"
#include "uci.h"

namespace Stockfish::Patricia {

// Get material count for a position
int get_material_count(const Position& pos) {
    constexpr int PieceValues[PIECE_TYPE_NB] = {
        0,    // NO_PIECE_TYPE
        100,  // PAWN
        320,  // KNIGHT
        330,  // BISHOP
        500,  // ROOK
        900,  // QUEEN
        0     // KING
    };

    int material = 0;
    for (PieceType pt = PAWN; pt <= QUEEN; ++pt) {
        material += popcount(pos.pieces(pt)) * PieceValues[pt];
    }
    return material;
}

// Get material advantage for side to move
int get_material_diff(const Position& pos) {
    constexpr int PieceValues[PIECE_TYPE_NB] = {
        0,    // NO_PIECE_TYPE
        100,  // PAWN
        320,  // KNIGHT
        330,  // BISHOP
        500,  // ROOK
        900,  // QUEEN
        0     // KING
    };

    Color us = pos.side_to_move();
    Color them = ~us;

    int our_material = 0;
    int their_material = 0;

    for (PieceType pt = PAWN; pt <= QUEEN; ++pt) {
        our_material += popcount(pos.pieces(us, pt)) * PieceValues[pt];
        their_material += popcount(pos.pieces(them, pt)) * PieceValues[pt];
    }

    return our_material - their_material;
}

void init_patricia_state(PatriciaState& state, const Position& pos) {
    state.phase = Eval::NNUE::PatriciaPhase::Middlegame;
    state.last_phase_check_depth = 0;
    state.starting_material = get_material_count(pos);
    state.history_length = 0;

    // Initialize accumulator
    auto& networks = Eval::NNUE::get_patricia_networks();
    networks.get_network(state.phase).init_accumulator(state.accumulator);
}

Eval::NNUE::PatriciaPhase detect_phase(Value eval, int depth) {
    // Patricia's phase detection (from search.h:1630-1636)
    // At depth 6:
    //   eval < -100 → Endgame
    //   eval > 300  → Sacrifice
    //   Otherwise   → Middlegame

    if (depth >= 6) {
        if (eval < VALUE_DRAW - 100)
            return Eval::NNUE::PatriciaPhase::Endgame;
        else if (eval > VALUE_DRAW + 300)
            return Eval::NNUE::PatriciaPhase::Sacrifice;
    }

    return Eval::NNUE::PatriciaPhase::Middlegame;
}

Value evaluate_patricia(const Position& pos,
                        PatriciaState& state,
                        int depth,
                        int search_ply) {
    auto& networks = Eval::NNUE::get_patricia_networks();

    // Check if phase needs updating
    if (depth >= 6 && depth != state.last_phase_check_depth) {
        // Get preliminary eval to detect phase
        Value prelim_eval = Value(networks.evaluate(state.accumulator, state.phase, pos.side_to_move() == WHITE));

        Eval::NNUE::PatriciaPhase new_phase = detect_phase(prelim_eval, depth);

        if (new_phase != state.phase) {
            state.phase = new_phase;
            // Reinitialize accumulator for new network
            networks.get_network(state.phase).init_accumulator(state.accumulator);
        }

        state.last_phase_check_depth = depth;
    }

    // Evaluate with current phase network
    Value eval = Value(networks.evaluate(state.accumulator, state.phase, pos.side_to_move() == WHITE));

    // Apply Patricia's aggressive modifiers
    eval = Modifiers::better_than_material(eval, pos);
    eval = Modifiers::sacrifice_bonus(pos, state, eval, search_ply);
    eval = Modifiers::material_scaling(eval, pos);

    return eval;
}

namespace Modifiers {

Value draw_contempt(const Position& pos, Value base_score) {
    // Patricia's draw contempt: ±50 based on material (search.h:536-546)
    int material_diff = get_material_diff(pos);

    Value contempt = VALUE_DRAW;
    if (material_diff < 0)
        contempt += 50;  // We're behind, accept draws
    else if (material_diff > 0)
        contempt -= 50;  // We're ahead, avoid draws

    return base_score + contempt;
}

Value sacrifice_bonus(const Position& pos,
                     const PatriciaState& state,
                     Value eval,
                     int search_ply) {
    // Patricia's asymmetric sacrifice bonus (search.h:136-173)
    // Only bonuses for the original side to move (checked via search_ply parity)

    bool our_side = (search_ply % 2 == 0);
    if (!our_side)
        return VALUE_ZERO;

    int total_material = get_material_count(pos);

    // Only apply in non-endgame positions
    if (total_material <= 3500)
        return VALUE_ZERO;

    // Detect sacrifices in search history
    // (Simplified version - full implementation would track entire history)
    int material_diff = get_material_diff(pos);
    int starting_diff = state.starting_material;

    // If we sacrificed material (material diff got worse)
    if (material_diff < starting_diff - 100) {
        // Bonus based on eval
        if (eval > VALUE_DRAW + 300)
            return Value(80);  // Big bonus when completely winning
        else if (eval > VALUE_DRAW)
            return Value(40);  // Small bonus when ahead
    }

    return VALUE_ZERO;
}

Value material_scaling(Value eval, const Position& pos) {
    // Patricia's material scaling (search.h:175-180)
    // Scale eval by material to discourage trading when ahead

    int total_material = get_material_count(pos);

    // multiplier = ((750 + total_material/25) / 1024)
    float multiplier = (750.0f + total_material / 25.0f) / 1024.0f;

    return Value(eval * multiplier);
}

Value better_than_material(Value eval, const Position& pos) {
    // Patricia's "better than material" bonus (search.h:126-133)
    // Currently commented out in Patricia but we can enable it

    int material_diff = get_material_diff(pos);
    int material_eval = material_diff;  // Rough material-only evaluation

    constexpr int threshold = 50;

    // Give bonus if position is much better than material suggests
    if (eval > 0 && eval > material_eval + threshold) {
        int bonus = 25 + (eval - material_eval - threshold) / 10;
        return Value(bonus);
    }
    else if (eval < 0 && eval < material_eval - threshold) {
        int bonus = -(25 + (material_eval - eval - threshold) / 10);
        return Value(bonus);
    }

    return VALUE_ZERO;
}

}  // namespace Modifiers

}  // namespace Stockfish::Patricia
