// Copyright (c) 2023-2026 Takehiro Yoshioka
// SPDX-License-Identifier: MIT

#ifndef SKIP_HPP
#define SKIP_HPP

#include <vector>
#include <utility>

#include "config.hpp"
#include "uctnode.hpp"
#include "digitalcurling/digitalcurling.hpp"
#include "apply_move.hpp"
#include "utility.hpp"

namespace dc = digitalcurling;

using namespace config;

namespace digitalcurling::client {

class Skip
{
    public:
        Skip();

        void OnInit(
            dc::Team const,
            GameRule const,
            dc::GameSetting const&,   
            std::unique_ptr<dc::simulators::ISimulatorFactory>,    
            std::vector<std::unique_ptr<dc::players::IPlayerFactory>> const&,
            int const
        );

        float search(UctNode*, int);
        void searchById(UctNode*, int, std::pair<int, int>);

        void updateNodes();
        void updateParent(UctNode*, float);

        void updateCount(UctNode*, std::pair<int, int>, int);

        void SimulateMove(UctNode*, std::pair<int, int>, int);
        std::pair<torch::Tensor, std::vector<float>> EvaluateGameState(std::vector<dc::GameState>, dc::GameSetting);
        void EvaluateQueue();

        dc::moves::Move command(const dc::GameState&);

    private:
        dc::Team team_;
        torch::jit::script::Module module_;
        GameRule game_rule_;
        dc::GameSetting game_setting_;
        std::array<std::unique_ptr<dc::simulators::ISimulator>, nLoop> simulators_;
        std::unique_ptr<simulators::ISimulator> simulator_;
        std::unique_ptr<dc::simulators::ISimulatorStorage> simulator_storage_;
        std::array<std::unique_ptr<dc::players::IPlayer>, 4> players_;
        std::vector<std::array<std::unique_ptr<dc::players::IPlayer>, 4>> players_pool_;
        std::chrono::milliseconds limit_;
        std::vector<std::vector<double>> win_table_;
        torch::Device device_;

        torch::ScalarType dtype_;

        std::vector<UctNode*> queue_evaluate_;
        std::vector<int> queue_simulate_;

        std::array<UctNode*, nLoop> queue_create_child_;
        std::array<std::pair<int, int>, nLoop> queue_create_child_index_;
        std::array<bool, nLoop> flag_create_child_;

        std::array<dc::GameState, nLoop> temp_game_states_;
        int simulator_pool_size_;

        torch::Tensor filt_;

        int kShotPerEnd_;

// torch::jit::script::Module module; // モデル

// // シミュレーション用変数
// dc::GameSetting g_game_setting;
// std::unique_ptr<dc::ISimulator> g_simulator;
// std::array<std::unique_ptr<dc::ISimulator>, nLoop> g_simulators;
// std::array<std::unique_ptr<dc::IPlayer>, 4> g_players;

// std::chrono::duration<double> limit; // 考慮時間制限




};

} // namespace digitalcurling::client

#endif // SKIP_HPP
