// Copyright (c) 2022-2026 UEC Takeshi Ito Laboratory
// SPDX-License-Identifier: MIT

#include <iostream>
#include <thread>

#include "digitalcurling/client/client_helpers.hpp"
#include "cnn_engine.hpp"
#include "digitalcurling/digitalcurling.hpp"

namespace digitalcurling::client {

// --- CNNEngine ---
CNNEngine::CNNEngine()
    : IStandardThinkingEngine()//,
        // IMixedThinkingEngine(),
        // IMixedDoublesThinkingEngine()
{
    skip_ = std::make_unique<Skip>();
}


std::vector<std::uint8_t> CNNEngine::OnInit(
    GameRule const& game_rule,
    GameSetting const& game_setting,
    std::unique_ptr<simulators::ISimulatorFactory> simulator,
    std::vector<std::unique_ptr<players::IPlayerFactory>> const& players
) {
    // auto sim = simulator->CreateSimulator();
    // auto inv_sim = dynamic_cast<simulators::InvertiblePluginSimulator*>(sim.get());
    // if (inv_sim == nullptr) {
    //     throw std::runtime_error("Simulator is not invertible simulator.");
    // }

    // sim.release();
    // simulator_ = std::move(std::unique_ptr<simulators::InvertiblePluginSimulator>(inv_sim));

    game_rule_ = game_rule;
    game_setting_ = game_setting;
    // store simulator and players for later initialization of curling-cnn
    simulator_factory_ = simulator->Clone();
    simulator_ = simulator->CreateSimulator();
    // save clones of player factories for forwarding to Skip::OnInit
    players_.clear();
    for (auto const& p : players) {
        players_.push_back(p ? p->Clone() : nullptr);
    }

    if (game_rule_.type == GameRuleType::kMixedDoubles) {
        kShotPerEnd = 5;
        return {0, 1};
    } else {
        kShotPerEnd = 8;
        return {0, 1, 2, 3};
    }

}

void CNNEngine::OnGameStart(
    Team const& team,
    std::vector<std::pair<GameState, std::optional<moves::Shot>>> states
) {
    team_ = team;
    // initialize curling-cnn Skip if available
    // skip_ = std::make_unique<Skip>();
    std::array<size_t, 4> player_order{0,1,2,3};
    try {
        std::vector<std::unique_ptr<dc::players::IPlayerFactory>> dc_players;
        for (auto const& p : players_) {
            dc_players.push_back(p ? p->Clone() : nullptr);
        }
        skip_->OnInit(team_, game_rule_, game_setting_, simulator_factory_->Clone(), std::move(dc_players), kShotPerEnd);
    } catch (...) {
        // ignore initialization failures and continue with fallback behavior
    }
}

void CNNEngine::OnNextEnd(GameState const& game_state) {
    // Nothing to do
}

// IMixedDoublesThinkingEngine::PositionedStoneOptions CNNEngine::OnDecidePositionedStone(GameState const& game_state) {
//     return PositionedStoneOptions::kCenterHouse;
// }

moves::Move CNNEngine::OnMyTurn(
    std::unique_ptr<players::IPlayerFactory> const& player_factory,
    GameState const& game_state,
    std::optional<moves::Shot> const& last_shot
) {
    // If curling-cnn is available and initialized, use it to decide the move
    if (skip_) {
        try {
                auto dc_move = skip_->command(game_state);
            if (std::holds_alternative<dc::moves::Shot>(dc_move)) {
                auto const& s = std::get<dc::moves::Shot>(dc_move);
                moves::Shot out(s.translational_velocity, s.angular_velocity, s.release_angle);
                return moves::Move(out);
            } else {
                // concede or other move: forward as concede
                return moves::Move(moves::Concede());
            }
        } catch (...) {
            // fallthrough to fallback below
        }
    }

    // fallback (simple default move)
    std::this_thread::sleep_for(std::chrono::microseconds(100));
    std::cout << "Shot decided (fallback)" << std::endl;
    return moves::Move(moves::Shot(2.4f, -1.57f, 1.6f));
}

void CNNEngine::OnOpponentTurn(GameState const& game_state,std::optional<moves::Shot> const& last_shot) {
    // Nothing to do
}

void CNNEngine::OnGameOver(GameState const& game_state) {
    // Nothing to do
}

} // namespace digitalcurling::client
