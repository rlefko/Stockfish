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
    state.starting_material_diff = get_material_diff(pos);  // Store material advantage, not total
    state.history_length = 0;

    // Initialize accumulator stack
    state.current = &state.accumulatorStack[0];
    auto& networks = Eval::NNUE::get_patricia_networks();
    const auto& network = networks.get_network(state.phase);

    // Initialize base accumulator and refresh from position
    network.init_accumulator(*state.current);
    Eval::NNUE::refresh_accumulator(*state.current, pos, network);
}

// Helper to apply a feature update to both perspectives
static void apply_feature_update(Eval::NNUE::PatriciaAccumulator& acc,
                                  const Eval::NNUE::PatriciaNetwork& network,
                                  Piece piece,
                                  Square square,
                                  bool add) {
    const auto [white_idx, black_idx] = Eval::NNUE::feature_indices(piece, square);

    // Update white perspective
    network.update_accumulator(acc, true, white_idx, add);

    // Update black perspective
    network.update_accumulator(acc, false, black_idx, add);
}

void push_patricia_accumulator(PatriciaState& state,
                                const Position& pos,
                                const DirtyPiece& dp) {
    auto& networks = Eval::NNUE::get_patricia_networks();
    const auto& network = networks.get_network(state.phase);

    // Copy parent accumulator to child position
    state.current[1] = state.current[0];
    state.current++;

    // Apply incremental updates based on DirtyPiece

    // 1. Remove piece from origin square (if not SQ_NONE)
    if (dp.from != SQ_NONE) {
        apply_feature_update(*state.current, network, dp.pc, dp.from, false);
    }

    // 2. Add piece to destination square (if not SQ_NONE, e.g., not a promotion)
    if (dp.to != SQ_NONE) {
        apply_feature_update(*state.current, network, dp.pc, dp.to, true);
    }

    // 3. Handle captures or castling rook removal
    if (dp.remove_sq != SQ_NONE) {
        apply_feature_update(*state.current, network, dp.remove_pc, dp.remove_sq, false);
    }

    // 4. Handle promotions or castling rook addition
    if (dp.add_sq != SQ_NONE) {
        apply_feature_update(*state.current, network, dp.add_pc, dp.add_sq, true);
    }
}

void pop_patricia_accumulator(PatriciaState& state) {
    state.current--;
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
        Value prelim_eval = Value(networks.evaluate(*state.current, state.phase, pos.side_to_move() == WHITE));

        Eval::NNUE::PatriciaPhase new_phase = detect_phase(prelim_eval, depth);

        if (new_phase != state.phase) {
            state.phase = new_phase;
            // Only refresh when switching networks
            const auto& new_network = networks.get_network(state.phase);
            Eval::NNUE::refresh_accumulator(*state.current, pos, new_network);
        }

        state.last_phase_check_depth = depth;
    }

    // Evaluate with current phase network (accumulator already updated incrementally)
    Value eval = Value(networks.evaluate(*state.current, state.phase, pos.side_to_move() == WHITE));

    // Guard: Don't modify mate or TB scores
    if (is_win(eval) || is_loss(eval))
        return eval;

    // Apply Patricia's aggressive modifiers
    Value bonus1 = Modifiers::better_than_material(eval, pos);
    Value bonus2 = Modifiers::sacrifice_bonus(pos, state, eval, search_ply);
    eval += bonus1 + bonus2;

    // Material scaling (discourage trades when ahead)
    eval = Modifiers::material_scaling(eval, pos);

    // Halfmove clock scaling (prevent 50-move draws)
    // Patricia's implementation: eval * (200 - halfmoves) / 200
    int halfmove_clock = pos.state()->rule50;
    if (halfmove_clock > 0) {
        eval = Value(eval * (200 - halfmove_clock) / 200);
    }

    // Draw contempt (±50cp based on material advantage)
    eval = Modifiers::draw_contempt(pos, eval);

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
    // Compare current material advantage to starting material advantage
    int material_diff = get_material_diff(pos);
    int starting_diff = state.starting_material_diff;  // Fixed: now comparing like units

    // If we sacrificed material (material advantage got worse by 100+ centipawns)
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
    // Increased threshold to reduce noise from frequent triggering

    int material_diff = get_material_diff(pos);
    int material_eval = material_diff;  // Rough material-only evaluation

    constexpr int threshold = 150;  // Increased from 50 to avoid excessive bonuses

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
