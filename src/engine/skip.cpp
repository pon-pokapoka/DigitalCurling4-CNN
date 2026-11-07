// Copyright (c) 2023-2026 Takehiro Yoshioka
// SPDX-License-Identifier: MIT

#include "skip.hpp"

#include <iostream>
#include <omp.h>
#include <tuple>
#include <queue>
#include <atomic>
#include <algorithm>
#include <thread>
#include <random>
#include <cstdlib>
#include <chrono>
#include <mutex>
#include <condition_variable>
#include <exception>
#include <stdexcept>
#include <numeric>
#include <map>
#include <fstream>
#include <unordered_set>

#include <torch/script.h>
#include <torch/cuda.h>
#include <torch/csrc/api/include/torch/nn/functional/activation.h>
#include <ATen/DeviceAccelerator.h>
#include <ATen/Parallel.h>

#include "readcsv.hpp"
#include "search_policy.hpp"
#include "executablepath.hpp"
#include <sstream>

namespace F = torch::nn::functional;

namespace {

struct SearchProfile {
    std::atomic<long long> total_ns{0};
    std::atomic<long long> ucb_ns{0};
    std::atomic<long long> topk_ns{0};
    std::atomic<long long> simulate_ns{0};
    std::atomic<long long> simulate_setup_ns{0};
    std::atomic<long long> apply_move_ns{0};
    std::atomic<long long> recurse_ns{0};
    std::atomic<long long> calls{0};
    std::atomic<long long> sim_calls{0};
    std::atomic<long long> recurse_calls{0};
};

SearchProfile g_search_profile;
constexpr bool kEnableLoopPerfLogging = false;

inline void ResetSearchProfile() {
    g_search_profile.total_ns.store(0, std::memory_order_relaxed);
    g_search_profile.ucb_ns.store(0, std::memory_order_relaxed);
    g_search_profile.topk_ns.store(0, std::memory_order_relaxed);
    g_search_profile.simulate_ns.store(0, std::memory_order_relaxed);
    g_search_profile.simulate_setup_ns.store(0, std::memory_order_relaxed);
    g_search_profile.apply_move_ns.store(0, std::memory_order_relaxed);
    g_search_profile.recurse_ns.store(0, std::memory_order_relaxed);
    g_search_profile.calls.store(0, std::memory_order_relaxed);
    g_search_profile.sim_calls.store(0, std::memory_order_relaxed);
    g_search_profile.recurse_calls.store(0, std::memory_order_relaxed);
}

inline void PrintSearchProfile() {
    long long total_ns = g_search_profile.total_ns.load(std::memory_order_relaxed);
    long long ucb_ns = g_search_profile.ucb_ns.load(std::memory_order_relaxed);
    long long topk_ns = g_search_profile.topk_ns.load(std::memory_order_relaxed);
    long long simulate_ns = g_search_profile.simulate_ns.load(std::memory_order_relaxed);
    long long simulate_setup_ns = g_search_profile.simulate_setup_ns.load(std::memory_order_relaxed);
    long long apply_move_ns = g_search_profile.apply_move_ns.load(std::memory_order_relaxed);
    long long recurse_ns = g_search_profile.recurse_ns.load(std::memory_order_relaxed);
    long long calls = g_search_profile.calls.load(std::memory_order_relaxed);
    long long sim_calls = g_search_profile.sim_calls.load(std::memory_order_relaxed);
    long long recurse_calls = g_search_profile.recurse_calls.load(std::memory_order_relaxed);

    auto to_ms = [](long long ns) {
        return static_cast<double>(ns) / 1'000'000.0;
    };
    auto pct = [total_ns](long long ns) {
        if (total_ns <= 0) return 0.0;
        return 100.0 * static_cast<double>(ns) / static_cast<double>(total_ns);
    };

    std::cout
        << "SearchProfile: calls=" << calls
        << " sim_calls=" << sim_calls
        << " recurse_calls=" << recurse_calls
        << " total=" << to_ms(total_ns) << "ms"
        << " ucb=" << to_ms(ucb_ns) << "ms(" << pct(ucb_ns) << "%)"
        << " topk=" << to_ms(topk_ns) << "ms(" << pct(topk_ns) << "%)"
        << " sim=" << to_ms(simulate_ns) << "ms(" << pct(simulate_ns) << "%)"
        << " sim_setup=" << to_ms(simulate_setup_ns) << "ms"
        << " apply_move=" << to_ms(apply_move_ns) << "ms"
        << " recurse=" << to_ms(recurse_ns) << "ms(" << pct(recurse_ns) << "%)"
        << std::endl;
}

int ResolveClientThreads() {
    if (const char* env = std::getenv("DC_CLIENT_THREADS"); env != nullptr) {
        int configured = std::atoi(env);
        if (configured > 0) {
            return configured;
        }
    }

    unsigned hw = std::thread::hardware_concurrency();
    int cores = hw > 0 ? static_cast<int>(hw) : omp_get_max_threads();
    return std::max(1, cores);
}

} // namespace

// int const policy_weight = 16;
// int const policy_width = 32;
// int const policy_rotation = 2;
namespace digitalcurling::client {

Skip::Skip() : module_(),
game_setting_(),
simulators_(),
players_(),
limit_(),
win_table_(),
device_(torch::kCPU),
dtype_(torch::kBFloat16),
queue_evaluate_(),
queue_simulate_(),
queue_create_child_(),
queue_create_child_index_(),
flag_create_child_(),
temp_game_states_(),
simulator_pool_size_(1),
legal_mask_(search_policy::LegalMask())
{
    int const client_threads = ResolveClientThreads();
    omp_set_dynamic(0);
    omp_set_num_threads(client_threads);
    at::set_num_threads(client_threads);
    at::set_num_interop_threads(1);

    #pragma omp parallel for
    for (auto i=0; i < nLoop; ++i) {
        flag_create_child_[i] = false;
    }

    // Retrieve and display the executable's absolute path
    std::filesystem::path dcRootPath;

    std::string execPath = getExecutablePath();
    if (execPath.empty()) {
        throw std::runtime_error("cannot determine the executable path, needed to find the model files");
    }

    std::cout << "Executable Path: " << execPath << std::endl;

    // If you want to work with the path, use std::filesystem
    std::filesystem::path execFilePath(execPath);
    std::cout << "Executable Directory: " << execFilePath.parent_path() << std::endl;

    dcRootPath = execFilePath.parent_path().parent_path().parent_path();

    // The win table and the model are required: a failure to load either stops the client.
    auto const modelDir = dcRootPath / "src" / "engine" / "model";
    win_table_ = readcsv(modelDir / "win_table.csv");
    // Rows are score differences -9..9; columns are the ends completed, 0..10.
    requireTableShape(win_table_, 19, 11, (modelDir / "win_table.csv").string());


    torch::NoGradGuard no_grad; 
    device_ = torch::kCPU;
    if (torch::cuda::is_available()) {
        std::cout << "CUDA is available!" << std::endl;
        device_ = torch::kCUDA;
    }   
    else {
        std::cout << "CUDA is not available." << std::endl;
    }

    dtype_ = torch::kBFloat16;

    try {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        std::cout << "model loading..." << std::endl;
        module_ = torch::jit::load(modelDir / "traced.pt", device_);
        module_.to(dtype_);
        std::cout << "model loaded" << std::endl;
    }
    catch (const c10::Error& e) {
        throw std::runtime_error("cannot load the model " + (modelDir / "traced.pt").string() + ": " + e.what_without_backtrace());
    }

    // ここでCNNによる推論を行うことで、次回以降の速度が早くなる
    // 使うバッチサイズすべてで行っておく
    std::cout << "initial inference\n";
    for (auto i = 0; i < 10; ++i) {
        std::cout << "." << std::flush;
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(torch::rand({nBatchSize, nChannel, height, width}, dtype_).to(device_));

        // Execute the model and turn its output into a tensor.
        auto outputs = module_.forward(inputs).toTuple();
        torch::Tensor out1 = outputs->elements()[0].toTensor().reshape({nBatchSize, policy_weight * policy_width * policy_rotation}).to(torch::kCPU);
    }
    std::cout << std::endl;
    for (auto i = 0; i < 10; ++i) {
        std::cout << "." << std::flush;
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(torch::rand({1, nChannel, height, width}, dtype_).to(device_));

        // Execute the model and turn its output into a tensor.
        auto outputs = module_.forward(inputs).toTuple();
        torch::Tensor out1 = outputs->elements()[0].toTensor().reshape({1, policy_weight * policy_width * policy_rotation}).to(torch::kCPU);
    }
    if (device_.is_cuda()) at::accelerator::emptyCache();
    std::cout << std::endl;
}


void Skip::OnInit(
    dc::Team const g_team,
    GameRule const game_rule,
    dc::GameSetting const& game_setting,
    std::unique_ptr<simulators::ISimulatorFactory> simulator,
    std::vector<std::unique_ptr<players::IPlayerFactory>> const& players,
    int const kShotPerEnd
) {
    team_ = g_team;

    simulator_ = simulator->CreateSimulator();
    simulator_pool_size_ = std::max(1, std::min(nLoop, ResolveClientThreads()));
    for (int i = 0; i < simulator_pool_size_; ++i) {
        simulators_[i] = simulator->CreateSimulator();
    }
    for (int i = simulator_pool_size_; i < nLoop; ++i) {
        simulators_[i].reset();
    }

    game_rule_ = game_rule;
    game_setting_ = game_setting;

    players_ = {
        players[0] ? players[0]->CreatePlayer() : nullptr,
        players[1] ? players[1]->CreatePlayer() : nullptr,
        players[2] ? players[2]->CreatePlayer() : nullptr,
        players[3] ? players[3]->CreatePlayer() : nullptr
    };

    players_pool_.clear();
    players_pool_.resize(simulator_pool_size_);
    for (int t = 0; t < simulator_pool_size_; ++t) {
        for (int p = 0; p < 4; ++p) {
            players_pool_[t][p] = players[p] ? players[p]->CreatePlayer() : nullptr;
        }
    }

    kShotPerEnd_ = kShotPerEnd;

    dc::GameState temp_game_state(game_setting_);

    // 考慮時間制限
    // ショット数で等分するが、超過分を考慮して0.9倍しておく
    limit_ = std::chrono::milliseconds(static_cast<long long>(game_setting_.thinking_time[team_].count() * 0.9 / (kShotPerEnd_) / game_setting_.max_end));

    // ショットシミュレーションの動作確認
    // しなくて良い
    // std::cout << "initial simulation\n";
    // for (auto j = 0; j < 10; ++j) {
    //     std::cout << "." << std::flush;
    //     dc::GameState dummy_game_state(g_game_setting);
    //     std::array<dc::GameState, nBatchSize> dummy_game_states; 
    //     std::array<dc::Move, nBatchSize> dummy_moves;
    //     auto & dummy_player = *g_players[0];
    //     dc::moves::Shot dummy_shot = {dc::Vector2(0, 2.5), dc::moves::Shot::Rotation::kCW};
    //     #pragma omp parallel for
    //     for (auto i=0; i < nBatchSize; ++i) {
    //         dummy_game_states[i] = dummy_game_state;
    //     }
    //     #pragma omp parallel for
    //     for (auto i=0; i < nBatchSize; ++i) {
    //         dummy_moves[i] = dummy_shot;
    //         g_simulators[i]->Load(*g_simulator_storage);

    //         dc::ApplyMove(g_game_setting, *g_simulators[i],
    //             dummy_player, dummy_game_states[i], dummy_moves[i], std::chrono::milliseconds(0));
    //     }
    // }
    // std::cout << std::endl;

    // filt = torch::from_blob(utility::createFilter().data(), {policy_rotation, policy_weight, policy_width}, torch::kBool);

    // std::cout << filt << std::endl;

    // std::cout << utility::createFilter() << std::endl;

    // for (auto i=0; i < policy_rotation; ++i){
    //     for (auto j=0; j < policy_weight; ++j) {
    //         for (auto k=0; k < policy_width; ++k) {
    //             std::cout << filt.index({i, j, k}).item<bool>();
    //         }
    //         std::cout << "\n";
    //     }
    //     std::cout << "\n\n";
    // }


    // for (auto i=0; i < policy_rotation; ++i){
    //     for (auto j=0; j < policy_weight; ++j) {
    //         for (auto k=0; k < policy_width; ++k) {
    //             std::cout << utility::createFilter()[utility::Id3d1d(i, j, k)];
    //         }
    //         std::cout << "\n";
    //     }
    //     std::cout << "\n\n";
    // }
}

float Skip::search(UctNode* current_node, int k)
{
    auto start_total = std::chrono::steady_clock::now();
    g_search_profile.calls.fetch_add(1, std::memory_order_relaxed);

    float result = current_node->GetValue();
    // output = evaluate(current_node);
    // set_policy_value(current_node, output);
    // set_filter(current_node)

    // current_node.getPolicy();
    // filt = current_node.getFilter();

    if (!((current_node->GetGameState().shot == 0) && (current_node->GetParent()))) {        
        thread_local std::mt19937 gen(std::random_device{}());
        std::uniform_real_distribution<> dis(0.0, 1.0);

        torch::Tensor policy = current_node->GetPolicy();

        // std::cout << ;
        // std::cout << static_cast<int>(currencurrent_node->GetVisitCount()t_node->GetGameState().end) << "  " << static_cast<int>(current_node->GetGameState().shot) << std::endl;

        auto start_ucb = std::chrono::steady_clock::now();

        bool const my_turn = current_node->GetGameState().GetNextTeam() == team_;
        bool const is_root = current_node->GetParent() == nullptr;

        // Root only: sometimes try the least-visited legal shot instead of the
        // PUCT choice, so shots the CNN rates low still get evaluated. Deeper
        // nodes use plain PUCT so random play doesn't leak into ancestor values.
        int explore_action = -1;
        if (is_root && dis(gen) < root_explore_rate) {
            explore_action = search_policy::LeastVisitedLegalAction(current_node->GetChildVisitCount(), legal_mask_, gen);
        }

        torch::Tensor ucb_value;
        if (explore_action < 0) {
            ucb_value = search_policy::PuctScore(current_node->GetChildVisitCount(), current_node->GetChildSumValue(),
                                                 policy, legal_mask_, my_turn, c_puct);
        }
        auto end_ucb = std::chrono::steady_clock::now();
        g_search_profile.ucb_ns.fetch_add(
            std::chrono::duration_cast<std::chrono::nanoseconds>(end_ucb - start_ucb).count(),
            std::memory_order_relaxed
        );


        // if (omp_get_thread_num() == 0) std::cout << torch::sum(q).item<float>() << torch::sum(policy).item<float>() << torch::sum(current_node->GetChildVisitCount()).item<int>();

        auto start_topk = std::chrono::steady_clock::now();
        int action = explore_action;
        if (action < 0) action = std::get<1>(torch::topk(ucb_value, 1)).index({0, 0}).item<int>();
        auto end_topk = std::chrono::steady_clock::now();
        g_search_profile.topk_ns.fetch_add(
            std::chrono::duration_cast<std::chrono::nanoseconds>(end_topk - start_topk).count(),
            std::memory_order_relaxed
        );

        // auto child_indices = current_node->GetChildIndices();

        // std::vector<int> matching_indices;
        // for (auto it = child_indices.begin(); it != child_indices.end(); ++it) {
        //     if (*it == indices.index({0, 0}).item<int>()) {
        //     matching_indices.push_back(std::distance(child_indices.begin(), it));
        //     }
        // }

        // std::vector<int>::iterator it;
        // if (!matching_indices.empty()) {
        //     // std::random_device rd;
        //     // std::mt19937 gen(rd());
        //     // std::uniform_int_distribution<> dis(0, matching_indices.size() - 1);
        //     // it = child_indices.begin() + matching_indices[dis(gen)];
        //     it = child_indices.begin() + matching_indices[current_node->GetChildVisitCount().index({0, indices.index({0, 0}).item<int>()}).item<int>() % matching_indices.size()];
        // } else {
        //     it = child_indices.end();
        // }

        // auto it = std::find(child_indices.begin(), child_indices.end(), indices.index({0, 0}).item<int>());
        
        // if (omp_get_thread_num() == 0) std::cout << indices.index({0, 0}).item<int>() << "  " << std::endl;


        int action_visit_count = current_node->GetChildVisitCount().index({0, action}).item<int>();
        bool is_last_shot = (current_node->GetGameState().shot + 1) >= kShotPerEnd_ * 2;
        int sim_slot = action_visit_count % nSimulation;

        if (is_last_shot) {
            sim_slot = current_node->AllocateNextChildIndex(action);
            // std::filesystem::create_directories("build");
            // std::ofstream slot_log("build/last_shot_slots.csv", std::ios::app);
            // if (slot_log) {
            //     slot_log << static_cast<int>(current_node->GetGameState().shot) << "," << action << "," << sim_slot << "\n";
            // }
        }

        // On the last shot every descent requests a new child, so the lookup isn't
        // needed; it would also race with AllocateNextChildIndex resizing the vector.
        UctNode* next_node = is_last_shot ? nullptr : current_node->GetChild(std::make_pair(action, sim_slot));

        // std::cout << action << "  " << sim_slot << "  " << next_node << std::endl;

        if (!(next_node) || (action_visit_count < expand_threshold) || is_last_shot) {
            auto start_sim = std::chrono::steady_clock::now();
            queue_create_child_[k] = current_node;
            queue_create_child_index_[k] = std::make_pair(action, sim_slot);
            flag_create_child_[k] = true;
            SimulateMove(current_node, queue_create_child_index_[k], k);
            auto end_sim = std::chrono::steady_clock::now();
            g_search_profile.sim_calls.fetch_add(1, std::memory_order_relaxed);
            g_search_profile.simulate_ns.fetch_add(
                std::chrono::duration_cast<std::chrono::nanoseconds>(end_sim - start_sim).count(),
                std::memory_order_relaxed
            );

        } else {
            
            if (next_node->GetEvaluated()) result = next_node->GetValue();
            // else {
            //     queue_create_child_[k] = current_node;
            //     queue_create_child_index_[k] = indices.index({0, 0}).item<int>();
            //     flag_create_child_[k] = true;
            // }

            // if (result != -1) updateNode(current_node, it - child_indices.begin(), result);

            if (next_node->GetEvaluated() & (next_node->GetGameState().shot > 0)) {
                auto start_recurse = std::chrono::steady_clock::now();
                result = search(next_node, k);
                auto end_recurse = std::chrono::steady_clock::now();
                g_search_profile.recurse_calls.fetch_add(1, std::memory_order_relaxed);
                g_search_profile.recurse_ns.fetch_add(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(end_recurse - start_recurse).count(),
                    std::memory_order_relaxed
                );
            }
        }
    }

    auto end_total = std::chrono::steady_clock::now();
    g_search_profile.total_ns.fetch_add(
        std::chrono::duration_cast<std::chrono::nanoseconds>(end_total - start_total).count(),
        std::memory_order_relaxed
    );
    return result;
}


void Skip::searchById(UctNode* current_node, int k, std::pair<int, int> indices)
{
    queue_create_child_[k] = current_node;
    queue_create_child_index_[k] = indices;
    flag_create_child_[k] = true;
    SimulateMove(current_node, queue_create_child_index_[k], k);

}


void Skip::updateParent(UctNode* node, float value)
{
    if (node->GetParent()) {
        node->GetParent()->SetValue(value);
        node->GetParent()->SetCount(1);


        node->GetParent()->SetChildCountValue(node->GetIndices().first, 1, value, false);

        updateParent(node->GetParent(), value);
    }
}


void Skip::updateNodes()
{
    for (auto& node: queue_evaluate_) {
        float value = node->GetValue();
        // node->SetValue(value);
        node->SetCount(1);
        updateParent(node, value);
    }
}


void Skip::updateCount(UctNode* node, std::pair<int, int> indices, int count)
{
    node->SetCount(count);

    bool const my_turn = node->GetGameState().GetNextTeam() == team_;
    node->SetChildCountValue(indices.first, count, search_policy::VirtualLossValue(my_turn, count), true);

    if (node->GetParent()) {
        updateCount(node->GetParent(), node->GetIndices(), count);
    }
}


void Skip::SimulateMove(UctNode* current_node, std::pair<int, int> indices, int k)
{
    auto start_setup = std::chrono::steady_clock::now();
    auto temp_move = moves::Move(PixelToShot(indices.first));

    temp_game_states_[k] = current_node->GetGameState();

    int sim_index = 0;
    if (omp_in_parallel()) {
        sim_index = omp_get_thread_num();
    } else if (k >= 0) {
        sim_index = k;
    }
    if (simulator_pool_size_ > 0) {
        sim_index %= simulator_pool_size_;
    }

    int player_index = temp_game_states_[k].shot / 4;
    dc::players::IPlayer* current_player = nullptr;
    if (sim_index >= 0 && sim_index < static_cast<int>(players_pool_.size())) {
        current_player = players_pool_[sim_index][player_index].get();
    }
    if (!current_player) {
        current_player = players_[player_index].get();
    }
    if (!current_player) {
        return;
    }

    auto& sim = (sim_index >= 0 && sim_index < nLoop && simulators_[sim_index])
        ? *simulators_[sim_index]
        : *simulator_;

    auto end_setup = std::chrono::steady_clock::now();
    g_search_profile.simulate_setup_ns.fetch_add(
        std::chrono::duration_cast<std::chrono::nanoseconds>(end_setup - start_setup).count(),
        std::memory_order_relaxed
    );

    auto start_apply = std::chrono::steady_clock::now();

    ApplyMove(
        game_rule_,
        game_setting_,
        sim,
        *current_player,
        temp_game_states_[k],
        temp_move,
        std::chrono::milliseconds(0)
    );

    auto end_apply = std::chrono::steady_clock::now();
    g_search_profile.apply_move_ns.fetch_add(
        std::chrono::duration_cast<std::chrono::nanoseconds>(end_apply - start_apply).count(),
        std::memory_order_relaxed
    );

}


std::pair<torch::Tensor, std::vector<float>> Skip::EvaluateGameState(std::vector<GameState> game_states, GameSetting game_setting)
{
    torch::NoGradGuard no_grad; 
   
    auto start = std::chrono::system_clock::now();
    auto now = std::chrono::system_clock::now();

    ModelInput model_input = GameStateToInput(game_states, game_setting, device_, dtype_);
    now = std::chrono::system_clock::now();
    if constexpr (kEnableLoopPerfLogging) {
        std::cout << "Evaluate: " << std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count() << " msec" << std::endl;
    }




    auto outputs = module_.forward(model_input.inputs).toTuple();
    now = std::chrono::system_clock::now();
    if constexpr (kEnableLoopPerfLogging) {
        std::cout << "Evaluate: " << std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count() << " msec" << std::endl;
    }

    int size = static_cast<int>(game_states.size());

    // Softmax in float on the device: bf16 on the CPU is slow and rounds close moves to ties.
    auto policy = F::softmax(outputs->elements()[0].toTensor().to(torch::kFloat).reshape({size, policy_weight * policy_width * policy_rotation}), 1)
                      .to(torch::kCPU).contiguous();

    std::vector<std::vector<float>> win_rate_array(size, std::vector<float>(kShotPerEnd_*2+1));

    // if (size==1) {
    //     auto sheet = model_input.inputs[0].toTensor().to(torch::kCPU);

    //     std::cout << sheet[0][0] << "\n" << sheet[0][1] << "\n";
    //     for (auto i=0; i<16; ++i){
    //         std::cout << sheet[0][i+2] << "\n";
    //     }
    // }
    // if (size!=1) std::cout << F::softmax(outputs, 1)[0] << std::endl;


    torch::Tensor score_prob = F::softmax(outputs->elements()[1].toTensor().to(torch::kFloat), 1).to(torch::kCPU).contiguous();
    float const* score_prob_data = score_prob.data_ptr<float>();
    int const score_stride = static_cast<int>(score_prob.size(1));
    std::vector<float> win_prob(size, 0);

    for (auto n=0; n < size; ++n){
        int next_end = 0;
        if (game_states[n].end >= game_setting_.max_end) {
            next_end = 10;
        } else {
            next_end = 10 - game_setting_.max_end + model_input.end[n] + 1;
        }
        if (game_states[n].shot % (kShotPerEnd_*2) == 0) {
            if (game_states[n].IsGameOver()) {
                win_prob[n] = team_ == game_states[n].game_result->winner;
            } else {
                int scorediff = model_input.score[n];
                if (scorediff > 9) scorediff = 9;
                else if (scorediff < -9) scorediff = -9;

                if (team_ == game_states[n].hammer) 
                win_prob[n] = win_table_[scorediff+9][next_end-1];
                else win_prob[n] = 1 - win_table_[scorediff+9][next_end-1];
            }
        } else {
            for (auto i=0; i < kShotPerEnd_*2+1; ++i){
                int scorediff_after_end = model_input.score[n] + i - kShotPerEnd_;
                if (scorediff_after_end > 9) scorediff_after_end = 9;
                else if (scorediff_after_end < -9) scorediff_after_end = -9;

                if (i > kShotPerEnd_) {
                    win_rate_array[n][i] = 1 - win_table_[9-scorediff_after_end][next_end];
                } else {
                    win_rate_array[n][i] = win_table_[scorediff_after_end+9][next_end];
                }
            }

            for (auto i=0; i < kShotPerEnd_*2+1; ++i) {
                win_prob[n] += score_prob_data[n * score_stride + i] * win_rate_array[n][i];
            }
            if (team_ != game_states[n].hammer) win_prob[n] = 1 - win_prob[n];
        }
    }
    now = std::chrono::system_clock::now();
    if constexpr (kEnableLoopPerfLogging) {
        std::cout << "Evaluate: " << std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count() << " msec" << std::endl;
    }

    // torch::Tensor win_rate = torch::from_blob(win_rate_array.data(), {size, kShotPerEnd+1});

    // torch::Tensor win_prob = at::sum(F::softmax(outputs, 1) * win_rate, 1).to(torch::kCPU);
    // now = std::chrono::system_clock::now();
    // std::cout << "Evaluate: " << std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count() << " msec" << std::endl;


    return std::make_pair(policy, win_prob);
}


void Skip::EvaluateQueue()
{       
    int size = static_cast<int>(queue_evaluate_.size());
    // std::cout << "Size: " << size << std::endl;

    std::vector<dc::GameState> game_states;
    // game_states.resize(size);

    // #pragma omp parallel for
    // for (auto i=0; i<size; ++i) {
    //     // std::cout << queue_evaluate[i]->GetIndices().first << "  " << queue_evaluate[i]->GetIndices().second << std::endl;
    //     game_states[i] = queue_evaluate[i]->GetGameState();
    // }

    for (auto i=0; i<size; ++i) {
        // std::cout << queue_evaluate[i]->GetIndices().first << "  " << queue_evaluate[i]->GetIndices().second << "  ";
        // std::cout << queue_evaluate[i] << std::endl;
        game_states.push_back(queue_evaluate_[i]->GetGameState());
    }

    // std::cout << "Get Game State" << std::endl;

    auto policy_value = EvaluateGameState(game_states, game_setting_);

    // std::cout << "Evaluate Game State" << std::endl;

    // torch::Tensor policy = torch::rand({size, policy_weight * policy_width * policy_rotation}).to(torch::kCPU);
    // torch::Tensor value = torch::rand({size});

    for (int i=0; i<size; ++i) {
        // std::cout << static_cast<int>(game_states[i].end) << static_cast<int>(game_states[i].shot) << std::endl;
        queue_evaluate_[i]->SetEvaluatedResults(policy_value.first.index({i}), policy_value.second[i]);
        // queue_evaluate[i]->SetFilter(utility::createFilter(game_states[i], g_game_setting));
    }

    // std::cout << "Set Evaluated Results" << std::endl;
}


dc::moves::Move Skip::command(dc::GameState const& game_state)
{
    // TODO AIを作る際はここを編集してください
    auto start = std::chrono::system_clock::now();

    // A search that failed on the previous shot may have left these set.
    queue_evaluate_.clear();
    flag_create_child_.fill(false);

    dc::GameState current_game_state = game_state;

    torch::NoGradGuard no_grad; 

    // 現在の局面を評価
    auto policy_value = EvaluateGameState({current_game_state}, game_setting_);


    // auto sheet = utility::GameStateToInput({current_game_state}, g_game_setting, torch::kCPU, dtype).inputs[0].toTensor();
    // // std::cout << sheet[0][0] << sheet[0][1] << std::endl;
    // for (auto i=0; i<16; ++i){
    //     std::cout << sheet[0][i+2][0][0].item<int>();
    // }
    // for (auto i=0; i < 2; ++i){
    //     for (auto j=0; j < utility::height; ++j) {
    //         for (auto k=0; k < utility::width; ++k) {
    //             std::cout << sheet.index({0, i, j, k}).item<float>();
    //         }
    //         std::cout << "\n";
    //     }
    //     std::cout << "\n";
    //     std::cout << "\n";
    // }

    // for (auto i=0; i < 16; ++i) {
    //     std::cout << sheet[0][i+2][0][0].item<float>();
    // }
    // auto policy = F::softmax(current_outputs.elements()[0].toTensor().reshape({1, 18700}).to(torch::kCPU), 1);

    // auto filt = utility::createFilter(current_game_state, g_game_setting);

    // root node
    std::unique_ptr<UctNode> root_node(new UctNode());
    root_node->SetGameState(current_game_state);
    root_node->SetEvaluatedResults(policy_value.first.index({0}), policy_value.second[0]);
    // root_node->SetFilter(filt);

    // for (auto i=0; i < policy_rotation; ++i){
    //     for (auto j=0; j < policy_weight; ++j) {
    //         for (auto k=0; k < policy_width; ++k) {
    //             std::cout << std::setw(2) << std::setfill('0') << std::setprecision(0) << static_cast<int>(policy_value.first.index({0, utility::Id3d1d(i, j, k)}).item<float>() * 10000) << " ";
    //         }
    //         std::cout << std::endl;
    //     }
    //     std::cout << std::endl;
    // }

    // std::cout << torch::rand({policy_rotation, policy_weight, policy_width}) * filt << std::endl;

    int count = 0;
    auto now = std::chrono::system_clock::now();
    auto a = std::chrono::system_clock::now();
    auto b = std::chrono::system_clock::now();
    while ((now - start < limit_)){
        ResetSearchProfile();
        ResetApplyMoveProfile();
        a = std::chrono::system_clock::now();
        // if (current_game_state.shot + 1 == kShotPerEnd) {
        // if (current_game_state.shot >= 0) {
        if (false) {
            if (count >= policy_rotation*policy_weight*policy_width*nSimulation) break;
            #pragma omp parallel for
            for (auto i = 0; i < nBatchSize; ++i) {
                searchById(root_node.get(), i, std::make_pair((count + i) % (policy_rotation*policy_weight*policy_width), (count + i) / (policy_rotation*policy_weight*policy_width)));

                if (flag_create_child_[i]) {
                    updateCount(queue_create_child_[i], queue_create_child_index_[i], virtual_loss);
                }
            }

        } else {
                {
                    int nthreads = ResolveClientThreads();
                    // std::cout << "Threads: " << nthreads << std::endl;
                    if (nthreads < 2) nthreads = std::max(2, omp_get_max_threads());
                    int updater_tid = nthreads - 1; // dedicate last thread to updates

                    std::queue<std::tuple<UctNode*, std::pair<int,int>, int>> update_queue;
                    std::mutex update_mutex;
                    std::condition_variable update_cv;
                    std::atomic<int> next_i(0);
                    // Children simulated this iteration. Only nBatchSize of them can be
                    // evaluated, so stop handing out descents once that many exist.
                    std::atomic<int> children_created(0);
                    std::atomic<int> workers_remaining(nthreads - 1);
                    // An exception must not leave the parallel region (that would
                    // terminate the process): keep the first one and rethrow it after.
                    std::exception_ptr search_error;
                    std::atomic<bool> search_failed(false);
                    auto const record_search_error = [&]() {
                        std::lock_guard<std::mutex> lock(update_mutex);
                        if (!search_error) search_error = std::current_exception();
                        search_failed.store(true);
                    };

                    // Each search() call runs torch tensor ops (UCB, torch::where, etc.).
                    // If ATen uses multiple threads inside each OpenMP worker we get N^2
                    // threads for N cores and oversubscription kills performance.
                    // Force ATen to single-threaded for the duration of the parallel block.
                    at::set_num_threads(1);

                    #pragma omp parallel num_threads(nthreads)
                    {
                        int tid = omp_get_thread_num();

                        if (tid == updater_tid) {
                            // updater thread: consume requests and call updateCount
                            try {
                                while (true) {
                                    std::unique_lock<std::mutex> lock(update_mutex);
                                    update_cv.wait(lock, [&]{ return !update_queue.empty() || workers_remaining.load() == 0; });
                                    while (!update_queue.empty()) {
                                        auto req = update_queue.front();
                                        update_queue.pop();
                                        lock.unlock();
                                        updateCount(std::get<0>(req), std::get<1>(req), std::get<2>(req));
                                        lock.lock();
                                    }
                                    if (workers_remaining.load() == 0 && update_queue.empty()) break;
                                }
                            } catch (...) {
                                record_search_error();
                            }
                        } else {
                            // worker threads: grab indices and run search
                            try {
                                while (true) {
                                    if (search_failed.load()) break;
                                    if (children_created.load(std::memory_order_relaxed) >= nBatchSize) break;
                                    int i = next_i.fetch_add(1);
                                    if (i >= nLoop) break;
                                    search(root_node.get(), i);

                                    if (flag_create_child_[i]) {
                                        children_created.fetch_add(1, std::memory_order_relaxed);
                                        std::unique_lock<std::mutex> lock(update_mutex);
                                        update_queue.emplace(queue_create_child_[i], queue_create_child_index_[i], virtual_loss);
                                        lock.unlock();
                                        update_cv.notify_one();
                                    }

                                }
                            } catch (...) {
                                record_search_error();
                            }

                            // worker finished its share; decrement remaining and notify updater if last.
                            // Taking the mutex keeps the notify from falling between the updater's
                            // predicate check and its wait, which would leave it asleep for good.
                            {
                                std::lock_guard<std::mutex> lock(update_mutex);
                                workers_remaining.fetch_sub(1);
                            }
                            update_cv.notify_one();
                        }
                    } // end parallel region

                    // Restore ATen thread count for evaluation (GPU batched inference).
                    at::set_num_threads(nthreads);

                    if (search_error) std::rethrow_exception(search_error);
                }
        }
        b = std::chrono::system_clock::now();
        if constexpr (kEnableLoopPerfLogging) {
            std::cout << "Search: " << std::chrono::duration_cast<std::chrono::milliseconds>(b - a).count() << " msec" << "\n";
            PrintSearchProfile();
            PrintApplyMoveProfile();
        }
        

        a = std::chrono::system_clock::now();
        for (auto i = 0; i < nLoop; ++i) {
            if (flag_create_child_[i]) {
                // Undo virtual loss applied after the parallel search phase.
                updateCount(queue_create_child_[i], queue_create_child_index_[i], -1*virtual_loss);

                // Link a child into the tree only if it is evaluated in this
                // iteration. An unevaluated child would occupy its slot for good:
                // later descents reaching it neither recurse nor back up a value.
                if (queue_evaluate_.size() < nBatchSize) {
                    queue_create_child_[i]->CreateChild(queue_create_child_index_[i]);
                    queue_create_child_[i]->GetChild(queue_create_child_index_[i])->SetGameState(temp_game_states_[i]);
                    queue_evaluate_.push_back(queue_create_child_[i]->GetChild(queue_create_child_index_[i]));
                    // std::cout << i << ":  ";
                    // std::cout << queue_create_child_index[i].first << "  " << queue_create_child_index[i].second << ";  ";
                    // std::cout << queue_create_child[i]->GetChild(queue_create_child_index[i])->GetIndices().first << "  " << queue_create_child[i]->GetChild(queue_create_child_index[i])->GetIndices().second << ";  ";
                    // std::cout << queue_create_child[i]->GetChild(queue_create_child_index[i]) << std::endl;
                }
            }
        }
        b = std::chrono::system_clock::now();
        if constexpr (kEnableLoopPerfLogging) {
            std::cout << "Expand: " << std::chrono::duration_cast<std::chrono::milliseconds>(b - a).count() << " msec" << "\n";
        }

        // for (auto ptr : queue_create_child) {
        //     ptr.reset();
        // }
        #pragma omp parallel for
        for (auto i=0; i < nLoop; ++i) {
            flag_create_child_[i] = false;
        }


        a = std::chrono::system_clock::now();

        count += queue_evaluate_.size();
        EvaluateQueue();
        b = std::chrono::system_clock::now();
        if constexpr (kEnableLoopPerfLogging) {
            std::cout << "Evaluate: " << std::chrono::duration_cast<std::chrono::milliseconds>(b - a).count() << " msec" << "\n";
        }

        a = std::chrono::system_clock::now();

        updateNodes();
        queue_evaluate_.clear();
        b = std::chrono::system_clock::now();
        if constexpr (kEnableLoopPerfLogging) {
            std::cout << "Update: " << std::chrono::duration_cast<std::chrono::milliseconds>(b - a).count() << " msec" << "\n";
        }


        // torch::Tensor ucb_value;
        // torch::Tensor q = torch::where(root_node->GetChildVisitCount() > 0,
        // root_node->GetChildSumValue() / root_node->GetChildVisitCount(), 0);
        // torch::Tensor u = root_node->GetPolicy() * std::sqrt(root_node->GetVisitCount()) / (1 + root_node->GetChildVisitCount());

        // ucb_value = q + u;
        // for (auto i=0; i < utility::policy_rotation; ++i){
        //     for (auto j=0; j < utility::policy_weight; ++j) {
        //         for (auto k=0; k < utility::policy_width; ++k) {
        //             std::cout << ucb_value.index({0, i*utility::policy_weight*utility::policy_width + j*utility::policy_width + k}).item<float>() << " ";
        //         }
        //         std::cout << "\n";
        //     }
        //     std::cout << "\n";
        //     std::cout << "\n";
        // }

        now = std::chrono::system_clock::now();
    }
    // GPUのキャッシュをクリア
    if (device_.is_cuda()) at::accelerator::emptyCache();

    auto msec = std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();
    std::cout << count << "  " << root_node->GetVisitCount() << " simulations in " << msec << " msec" << std::endl;

    // // If this is the last shot, dump a snapshot of the tree (game state, policy and child stats)
    // if (current_game_state.shot >= kShotPerEnd - 1) {
    //     auto DumpGameState = [&](dc::GameState const& gs, std::ostream &os) {
    //         os << "GameState: end=" << static_cast<int>(gs.end) << " shot=" << static_cast<int>(gs.shot)
    //            << " hammer=" << static_cast<int>(gs.hammer);
    //         os << " scores=[";
    //         for (int t = 0; t < 2; ++t) {
    //             os << "{";
    //             for (size_t e = 0; e < gs.scores[t].size(); ++e) {
    //                 if (gs.scores[t][e].has_value()) os << static_cast<int>(gs.scores[t][e].value());
    //                 else os << "-";
    //                 if (e + 1 < gs.scores[t].size()) os << ",";
    //             }
    //             os << "}";
    //             if (t == 0) os << ",";
    //         }
    //         os << "]\n";
    //     };

    //     std::function<void(UctNode*, std::ostream&, int)> DumpNode;
    //     DumpNode = [&](UctNode* node, std::ostream &os, int depth) {
    //         if (!node) return;
    //         os << "Node indices=(" << node->GetIndices().first << "," << node->GetIndices().second << ") visits=" << node->GetVisitCount();
    //         os << " value=" << node->GetValue() << " countValue=" << node->GetCountValue() << "\n";

    //         // policy
    //         // try {
    //         //     auto policy = node->GetPolicy();
    //         //     os << "  policy sum=" << policy.sum().item<float>() << " topk: ";
    //         //     int k = std::min(6, static_cast<int>(policy.numel()));
    //         //     auto topk = std::get<1>(torch::topk(policy, k));
    //         //     for (int i = 0; i < k; ++i) os << topk.index({0, i}).item<int>() << " ";
    //         //     os << "\n";
    //         // } catch (...) {os << "  policy not available\n";}

    //         // child tensors
    //         try {
    //             auto visit_counts = node->GetChildVisitCount();
    //             auto sum_values = node->GetChildSumValue();
    //             auto max_values = node->GetChildMaxValue();
    //             auto min_values = node->GetChildMinValue();
    //             int numActions = policy_weight * policy_width * policy_rotation;
    //             for (int a = 0; a < numActions; ++a) {
    //                 int visits = visit_counts.index({0, a}).item<int>();
    //                 if (visits == 0) {
    //                     // os << "  action=" << a << " visits=" << "0\n";
    //                     continue;
    //                 }
    //                 float meanq = sum_values.index({0, a}).item<float>() / visits;
    //                 float maxv = max_values.index({0, a}).item<float>();
    //                 float minv = min_values.index({0, a}).item<float>();
    //                 os << "  action=" << a << " visits=" << visits << " mean=" << meanq << " max=" << maxv << " min=" << minv << "\n";
    //             }
    //         } catch (...) {os << "  child stats not available\n";}

    //         if (depth <= 0) return;
    //         auto addresses = node->GetChildNodeAddresses();
    //         int numActions = policy_weight * policy_width * policy_rotation;
    //         for (int a = 0; a < numActions; ++a) {
    //             for (int s = 0; s < nSimulation; ++s) {
    //                 UctNode* child = addresses[a][s];
    //                 if (child) DumpNode(child, os, depth - 1);
    //             }
    //         }
    //     };

    //     std::ostringstream oss;
    //     oss << "=== LAST SHOT TREE SNAPSHOT ===\n";
    //     DumpGameState(current_game_state, oss);
    //     DumpNode(root_node.get(), oss, 3);
    //     oss << "=== END SNAPSHOT ===\n";
    //     std::cout << oss.str();
    // }

    // std::vector<int> child_indices = root_node->GetChildIndices();
    // std::vector<float> values;
    // std::array<std::array<std::array<int, policy_width>, policy_weight>, policy_rotation> win_rate;
    // std::fill(&win_rate[0][0][0], &win_rate[0][0][0] + policy_rotation * policy_weight * policy_width, 0);
    // if (current_game_state.shot + 1 == kShotPerEnd) {
        // values.resize(policy_width*policy_weight*policy_rotation, 0);
        // for (auto i=0; i < child_indices.size(); ++i) {
            // values[child_indices[i]] += root_node->GetChildById(child_indices[i])->GetCountValue() / nSimulation;

            // win_rate[child_indices[i] / (policy_width*policy_weight)][(child_indices[i] % (policy_width*policy_weight)) / policy_width][child_indices[i] % policy_width] += root_node->GetChildById(child_indices[i])->GetVisitCount();
        // }
        // for (auto i=0; i < policy_rotation; ++i){
        //     for (auto j=0; j < policy_weight; ++j) {
        //         for (auto k=0; k < policy_width; ++k) {
        //             win_rate[i][j][k] = values[utility::Id3d1d(i, j, k)];
        //         }GetCountValue
        //     }
        // }
    // } else {
        // for (auto index: child_indices) {
            // values.push_back(root_node->GetChild(index)->GetCountValue());
            // win_rate[index / (policy_width*policy_weight)][(index % (policy_width*policy_weight)) / policy_width][index % policy_width] += root_node->GetChild(index)->GetVisitCount();
        // }
        // for (auto index: child_indices) {
        //     win_rate[index / (policy_weight * policy_width)][index % (policy_weight * policy_width) / policy_width][index % (policy_weight * policy_width) % policy_width] = root_node->GetChild(index)->GetCountValue();
        // }

    // }


    // for (auto i=0; i < policy_rotation; ++i){
    //     // for (auto j=0; j < policy_weight; ++j) {
    //         int j = 0;
    //         for (auto k=0; k < policy_width; ++k) {
    //             std::cout << std::setw(2) << std::setfill('0') << std::setprecision(0) << static_cast<int>(win_rate[i][j][k] * 100) << " ";
    //         }
    //         std::cout << "\n";
    //     // }
    //     std::cout << "\n";
    //     std::cout << "\n";
    // }

    // if (current_game_state.shot % 2 == 0) {
    //     for (auto i=0; i < values.size(); ++i) {
    //         values[i] = 1 - values[i];
    //     }
    // }

    // for (auto i=0; i < policy_rotation; ++i){
    //     for (auto j=0; j < policy_weight; ++j) {
    //         for (auto k=0; k < policy_width; ++k) {
    //             std::cout << std::setw(2) << std::setfill('0') << std::setprecision(0) << static_cast<int>((torch::where(root_node->GetChildVisitCount() > 0, root_node->GetChildSumValue() / root_node->GetChildVisitCount(), 0) + policy_value.first).index({0, Id3d1d(i, j, k)}).item<float>() * 100) << " ";
    //             // std::cout << std::setw(2) << std::setfill('0') << std::setprecision(0) << static_cast<int>(root_node->GetChildVisitCount().index({0, Id3d1d(i, j, k)}).item<int>()) << " ";
    //         }
    //         std::cout << std::endl;
    //     }
    //     std::cout << std::endl;
    // }



    // int pixel_id = child_indices[static_cast<int>(std::distance(values.begin(), std::max_element(values.begin(), values.end())))];


    // 1. Calculate the mean Q-value (Winning Rate Estimate) for visited nodes.
    // Unvisited nodes will have a value of 0, but will be filtered later by the max_q check.
    int const visit_threshold = search_policy::FinalVisitThreshold(root_node->GetChildVisitCount().max().item<int>());
    torch::Tensor mean_q_values = torch::where(
        root_node->GetChildVisitCount() >= visit_threshold, 
        root_node->GetChildSumValue() / root_node->GetChildVisitCount() + 
        root_node->GetChildMinValue(),
        0 // Assign a very low value (or negative infinity) to unvisited/unreliable nodes
    );

    // 2. Find the Maximum Q-Value (Q_max)
    torch::Tensor max_q = mean_q_values.max();

    // 4. Identify the Plateau: Create a mask for actions where Q-value is within epsilon of Q_max
    torch::Tensor plateau_mask = (mean_q_values >= (max_q - tie_breaker_epsilon));

    // 5. Apply Policy Tie-Breaker:
    //    - For actions on the plateau, use their Policy Value (P) as the score.
    //    - For actions not on the plateau, assign a very low value (or negative infinity).
    torch::Tensor policy = root_node->GetPolicy(); // Assuming this is a tensor of P(a|s)
    torch::Tensor tie_breaker_score = torch::where(
        plateau_mask & legal_mask_,
        policy,
        0
    );

    // 6. Argmax on the Tie-Breaker Score:
    //    - torch::topk finds the maximum score.
    //    - .index({0, 0}).item<int>() extracts the index (pixel_id) of that maximum score.
    int pixel_id = std::get<1>(torch::topk(tie_breaker_score, 1)).index({0, 0}).item<int>();

    // std::cout << pixel_id << ":   " << pixel_id / (policy_weight * policy_width) << ", " << pixel_id % (policy_weight * policy_width) / policy_width << ", " << pixel_id % (policy_weight * policy_width) % policy_width << std::endl;

    // std::cout << std::accumulate(std::begin(values), std::end(values), 0.f) << std::endl;
    // if (std::accumulate(std::begin(values), std::end(values), 0.f) < 1e-6f) move = dc::moves::Concede();

    // move = dc::moves::Concede();

    // wait for 1 second
    // std::this_thread::sleep_for(std::chrono::seconds(1));

    // // Traverse tree to collect per-action child counts per shot and save to CSV
    // {
    //     const int numActions = policy_weight * policy_width * policy_rotation;
    //     std::map<int, std::vector<int>> max_children; // shot -> vector[action] = max children seen for that shot/action
    //     std::queue<UctNode*> qn;
    //     std::unordered_set<UctNode*> visited;
    //     qn.push(root_node.get());
    //     visited.insert(root_node.get());

    //     while (!qn.empty()) {
    //         UctNode* node = qn.front(); qn.pop();
    //         int shot = static_cast<int>(node->GetGameState().shot);
    //         auto const& addrs = node->GetChildNodeAddresses();

    //         if (!max_children.count(shot)) max_children[shot] = std::vector<int>(numActions, 0);

    //         for (int a = 0; a < numActions; ++a) {
    //             int children = 0;
    //             if (a < static_cast<int>(addrs.size())) {
    //                 for (auto ptr : addrs[a]) {
    //                     if (ptr) {
    //                         ++children;
    //                         if (visited.insert(ptr).second) qn.push(ptr);
    //                     }
    //                 }
    //             }
    //             max_children[shot][a] = std::max(max_children[shot][a], children);
    //         }
    //     }

    //     // write CSV snapshot of the current tree
    //     std::filesystem::create_directories("build");
    //     std::ofstream ofs("build/child_counts.csv", std::ofstream::app);
    //     if (ofs) {
    //         ofs << "shot,action,max_children\n";
    //         for (auto const& [shot, vec] : max_children) {
    //             for (int a = 0; a < static_cast<int>(vec.size()); ++a) {
    //                 ofs << shot << "," << a << "," << vec[a] << "\n";
    //             }
    //         }
    //         ofs.close();
    //     }
    // }

    return dc::moves::Move(PixelToShot(pixel_id));
}

} // namespace digitalcurling::client
