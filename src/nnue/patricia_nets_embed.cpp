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

// Embed Patricia's NNUE networks into the binary

#include "../incbin/incbin.h"
#include "nnue_patricia.h"

namespace Stockfish::Eval::NNUE {

// Embed the three Patricia networks
INCBIN(patricia_firefly, "patricia_nets/firefly.nnue");
INCBIN(patricia_rw3, "patricia_nets/rw3.nnue");
INCBIN(patricia_allie, "patricia_nets/allie.nnue");

// Global Patricia networks instance
static PatriciaNetworks g_patricia_networks;
static bool g_patricia_initialized = false;

// Initialize Patricia networks from embedded data
void init_patricia_networks() {
    if (!g_patricia_initialized) {
        g_patricia_networks.load_from_embedded(
            gpatricia_fireflyData,
            gpatricia_rw3Data,
            gpatricia_allieData
        );
        g_patricia_initialized = true;
    }
}

// Get the global Patricia networks instance
PatriciaNetworks& get_patricia_networks() {
    if (!g_patricia_initialized) {
        init_patricia_networks();
    }
    return g_patricia_networks;
}

}  // namespace Stockfish::Eval::NNUE
