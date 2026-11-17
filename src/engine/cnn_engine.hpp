// Copyright (c) 2022-2026 UEC Takeshi Ito Laboratory
// Copyright (c) 2023-2026 Takehiro Yoshioka
// SPDX-License-Identifier: MIT
#ifndef CNN_ENGINE_HPP
#define CNN_ENGINE_HPP

#pragma once

#include "digitalcurling/client/i_factory_creator.hpp"
#include "digitalcurling/client/i_thinking_engine.hpp"
#include "skip.hpp"

namespace digitalcurling::client {

class CNNEngine :
    public IStandardThinkingEngine //, public IMixedThinkingEngine , public IMixedDoublesThinkingEngine
{
public:
    CNNEngine();

    virtual inline std::string GetName() const override {
        return "cnn";
    }

    virtual std::vector<std::uint8_t> OnInit(
        GameRule const& game_rule,
        GameSetting const& game_setting,
        std::unique_ptr<simulators::ISimulatorFactory> simulator,
        std::vector<std::unique_ptr<players::IPlayerFactory>> const& players
    ) override;

    virtual void OnGameStart(
        Team const& team,
        std::vector<std::pair<digitalcurling::GameState, std::optional<moves::Shot>>> states
    ) override;
    virtual void OnNextEnd(GameState const& game_state) override;
    // PositionedStoneOptions OnDecidePositionedStone(GameState const& game_state) override;

    virtual moves::Move OnMyTurn(
        std::unique_ptr<players::IPlayerFactory> const& player_factory,
        GameState const& game_state,
        std::optional<moves::Shot> const& last_shot
    ) override;

    virtual void OnOpponentTurn(
        GameState const& game_state,
        std::optional<moves::Shot> const& last_shot
    ) override;

    virtual void OnGameOver(GameState const& game_state) override;

private:
    Team team_;
    GameRule game_rule_;
    GameSetting game_setting_;
    std::unique_ptr<Skip> skip_;
    std::unique_ptr<simulators::ISimulator> simulator_;
    std::unique_ptr<simulators::ISimulatorFactory> simulator_factory_;
    std::vector<std::unique_ptr<players::IPlayerFactory>> players_;
    // std::unique_ptr<simulators::InvertiblePluginSimulator> simulator_;

    size_t kShotPerEnd;
};

} // namespace digitalcurling::client

#endif // CNN_ENGINE_HPP