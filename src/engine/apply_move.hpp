// Copyright (c) 2023-2026 Takehiro Yoshioka
// SPDX-License-Identifier: MIT

#ifndef APPLY_MOVE_HPP
#define APPLY_MOVE_HPP

#include "digitalcurling/digitalcurling.hpp"
#include <chrono>
#include <optional>


using namespace digitalcurling;
using namespace digitalcurling::moves;
using namespace digitalcurling::players;
using namespace digitalcurling::simulators;

namespace digitalcurling::client {

void ResetApplyMoveProfile();
void PrintApplyMoveProfile();

GameState ApplyMove(
    GameRule const& ,
    GameSetting const& ,
    ISimulator& ,
    IPlayer& ,
    GameState& ,
    Move const& ,
    std::chrono::milliseconds 
);

} // namespace digitalcurling::client

#endif // APPLY_MOVE_HPP
