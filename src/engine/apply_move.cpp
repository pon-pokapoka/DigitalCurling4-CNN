#include "apply_move.hpp"

#include <atomic>
#include <chrono>
#include <iostream>
#include <vector>

namespace {

using namespace digitalcurling;
using namespace digitalcurling::moves;
using namespace digitalcurling::simulators;

struct ApplyMoveProfile {
    std::atomic<long long> calls{0};
    std::atomic<long long> total_ns{0};
    std::atomic<long long> play_ns{0};
    std::atomic<long long> convert_to_sim_ns{0};
    std::atomic<long long> set_stones_ns{0};
    std::atomic<long long> step_loop_ns{0};
    std::atomic<long long> get_convert_back_ns{0};
    std::atomic<long long> validate_update_ns{0};
    std::atomic<long long> step_calls{0};
};

ApplyMoveProfile g_apply_move_profile;

inline void AddNs(std::atomic<long long>& target, std::chrono::steady_clock::time_point begin, std::chrono::steady_clock::time_point end) {
    target.fetch_add(
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count(),
        std::memory_order_relaxed
    );
}

ISimulator::AllStones ConvertToSimulatorStones(StoneCoordinate const& stones) {
    ISimulator::AllStones all_stones;
    auto flat = stones.GetAllStones();

    for (size_t i = 0; i < flat.size(); ++i) {
        if (!flat[i].has_value()) {
            all_stones[i] = std::nullopt;
            continue;
        }

        all_stones[i] = ISimulator::StoneState(
            flat[i]->position,
            flat[i]->angle,
            {0.f, 0.f},
            0.f
        );
    }

    return all_stones;
}

StoneCoordinate ConvertToStoneCoordinate(ISimulator::AllStones const& all_stones) {
    StoneCoordinate result;

    for (size_t i = 0; i < all_stones.size(); ++i) {
        size_t idx = i % 8;
        auto& team_stones = (i < 8) ? result.team0 : result.team1;

        if (all_stones[i].has_value()) {
            team_stones[idx] = Stone(all_stones[i]->position, all_stones[i]->angle);
        } else {
            team_stones[idx].reset();
        }
    }

    return result;
}

bool IsValidStone(Stone const& stone, float sheet_width) {
    return stone.position.x + Stone::kRadius < sheet_width / 2.f
        && stone.position.x - Stone::kRadius > -sheet_width / 2.f
        && stone.position.y - Stone::kRadius < coordinate::kBackLineY
        && stone.position.y - Stone::kRadius > coordinate::kHogLineY;
}

bool IsValidMovingStone(Stone const& stone, float sheet_width) {
    return stone.position.x + Stone::kRadius < sheet_width / 2.f
        && stone.position.x - Stone::kRadius > -sheet_width / 2.f
        && stone.position.y - Stone::kRadius < coordinate::kBackLineY
        && stone.position.y + Stone::kRadius > 0.f;
}

bool ValidateShot(
    GameRule const& game_rule,
    std::uint8_t shot_in_end,
    Team deliver,
    StoneCoordinate const& before,
    StoneCoordinate const& after
) {
    return !game_rule.VerifyShot(shot_in_end, deliver, before, after).has_value();
}

void FinalizeEnd(GameSetting const& game_setting, GameState& game_state) {
    std::vector<StoneIndex> sorted = game_state.stones.GetSortedIndex();

    std::uint8_t score0 = 0;
    std::uint8_t score1 = 0;
    if (!sorted.empty()) {
        auto const& nearest_stone = game_state.stones.GetStone(sorted.front());
        if (nearest_stone.has_value() && nearest_stone->IsInHouse()) {
            Team scoring_team = sorted.front().team;
            for (auto const& idx : sorted) {
                auto const& stone = game_state.stones.GetStone(idx);
                if (!stone.has_value() || !stone->IsInHouse()) {
                    break;
                }
                if (idx.team != scoring_team) {
                    break;
                }
                if (scoring_team == Team::k0) {
                    ++score0;
                } else {
                    ++score1;
                }
            }
        }
    }

    game_state.scores.team0[game_state.end] = score0;
    game_state.scores.team1[game_state.end] = score1;

    if (score0 > 0) {
        game_state.hammer = Team::k1;
    } else if (score1 > 0) {
        game_state.hammer = Team::k0;
    }

    game_state.stones = StoneCoordinate{};
    game_state.end++;
    game_state.shot = 0;

    if (game_state.end < game_setting.max_end) {
        return;
    }

    int total0 = 0;
    int total1 = 0;
    for (size_t e = 0; e < game_state.scores[Team::k0].size(); ++e) {
        if (game_state.scores[Team::k0][e].has_value()) {
            total0 += *game_state.scores[Team::k0][e];
        }
        if (game_state.scores[Team::k1][e].has_value()) {
            total1 += *game_state.scores[Team::k1][e];
        }
    }

    if (total0 > total1) {
        game_state.game_result = GameResult{Team::k0, GameResult::Reason::kScore};
    } else if (total1 > total0) {
        game_state.game_result = GameResult{Team::k1, GameResult::Reason::kScore};
    } else {
        game_state.game_result = GameResult{Team::kInvalid, GameResult::Reason::kDraw};
    }
}

} // namespace

namespace digitalcurling::client {

void ResetApplyMoveProfile() {
    g_apply_move_profile.calls.store(0, std::memory_order_relaxed);
    g_apply_move_profile.total_ns.store(0, std::memory_order_relaxed);
    g_apply_move_profile.play_ns.store(0, std::memory_order_relaxed);
    g_apply_move_profile.convert_to_sim_ns.store(0, std::memory_order_relaxed);
    g_apply_move_profile.set_stones_ns.store(0, std::memory_order_relaxed);
    g_apply_move_profile.step_loop_ns.store(0, std::memory_order_relaxed);
    g_apply_move_profile.get_convert_back_ns.store(0, std::memory_order_relaxed);
    g_apply_move_profile.validate_update_ns.store(0, std::memory_order_relaxed);
    g_apply_move_profile.step_calls.store(0, std::memory_order_relaxed);
}

void PrintApplyMoveProfile() {
    long long calls = g_apply_move_profile.calls.load(std::memory_order_relaxed);
    long long total_ns = g_apply_move_profile.total_ns.load(std::memory_order_relaxed);
    long long play_ns = g_apply_move_profile.play_ns.load(std::memory_order_relaxed);
    long long convert_to_sim_ns = g_apply_move_profile.convert_to_sim_ns.load(std::memory_order_relaxed);
    long long set_stones_ns = g_apply_move_profile.set_stones_ns.load(std::memory_order_relaxed);
    long long step_loop_ns = g_apply_move_profile.step_loop_ns.load(std::memory_order_relaxed);
    long long get_convert_back_ns = g_apply_move_profile.get_convert_back_ns.load(std::memory_order_relaxed);
    long long validate_update_ns = g_apply_move_profile.validate_update_ns.load(std::memory_order_relaxed);
    long long step_calls = g_apply_move_profile.step_calls.load(std::memory_order_relaxed);

    auto to_ms = [](long long ns) {
        return static_cast<double>(ns) / 1'000'000.0;
    };
    auto pct = [total_ns](long long ns) {
        if (total_ns <= 0) return 0.0;
        return 100.0 * static_cast<double>(ns) / static_cast<double>(total_ns);
    };

    std::cout
        << "ApplyMoveProfile: calls=" << calls
        << " steps=" << step_calls
        << " total=" << to_ms(total_ns) << "ms"
        << " play=" << to_ms(play_ns) << "ms(" << pct(play_ns) << "%)"
        << " to_sim=" << to_ms(convert_to_sim_ns) << "ms(" << pct(convert_to_sim_ns) << "%)"
        << " set_stones=" << to_ms(set_stones_ns) << "ms(" << pct(set_stones_ns) << "%)"
        << " step_loop=" << to_ms(step_loop_ns) << "ms(" << pct(step_loop_ns) << "%)"
        << " get_back=" << to_ms(get_convert_back_ns) << "ms(" << pct(get_convert_back_ns) << "%)"
        << " validate=" << to_ms(validate_update_ns) << "ms(" << pct(validate_update_ns) << "%)"
        << std::endl;
}

GameState ApplyMove(
    GameRule const& game_rule,
    GameSetting const& game_setting,
    ISimulator& simulator,
    IPlayer& player,
    GameState& game_state,
    Move const& move,
    std::chrono::milliseconds time_limit
) {
    auto total_begin = std::chrono::steady_clock::now();
    g_apply_move_profile.calls.fetch_add(1, std::memory_order_relaxed);

    if (game_state.IsGameOver()) {
        AddNs(g_apply_move_profile.total_ns, total_begin, std::chrono::steady_clock::now());
        return game_state;
    }
    // std::cout << "In Game,    ";

    // Handle Concede
    if (std::holds_alternative<Concede>(move)) {
        game_state.game_result = GameResult{GetOpponentTeam(game_state.GetNextTeam()), GameResult::Reason::kConcede};
        AddNs(g_apply_move_profile.total_ns, total_begin, std::chrono::steady_clock::now());
        return game_state;
    }
    // std::cout << "Not Concede, " << std::endl;

    Team deliver_team = game_state.GetNextTeam();
    std::uint8_t shot_in_end = game_state.shot;
    StoneCoordinate before_stones = game_state.stones;

    // std::cout << "Deliver team: " << ((deliver_team == Team::k0) ? "Team 0" : "Team 1") << ", Shot in end: " << static_cast<int>(shot_in_end) << std::endl;

    const Shot& requested_shot = std::get<Shot>(move);
    auto play_begin = std::chrono::steady_clock::now();
    Shot const played_shot = player.Play(requested_shot);
    AddNs(g_apply_move_profile.play_ns, play_begin, std::chrono::steady_clock::now());

    // std::cout << "Requested shot: translational_velocity=" << requested_shot.translational_velocity
    //           << ", angular_velocity=" << requested_shot.angular_velocity
    //           << ", release_angle=" << requested_shot.release_angle << std::endl;
    // std::cout << "Played shot: translational_velocity=" << played_shot.translational_velocity
    //           << ", angular_velocity=" << played_shot.angular_velocity
    //           << ", release_angle=" << played_shot.release_angle << std::endl;


    auto to_sim_begin = std::chrono::steady_clock::now();
    ISimulator::AllStones all_stones = ConvertToSimulatorStones(before_stones);
    AddNs(g_apply_move_profile.convert_to_sim_ns, to_sim_begin, std::chrono::steady_clock::now());

    auto set_begin = std::chrono::steady_clock::now();
    simulator.SetStones(all_stones);
    AddNs(g_apply_move_profile.set_stones_ns, set_begin, std::chrono::steady_clock::now());

    // Set the moving stone's initial position, velocity and angular velocity
    std::uint8_t stone_idx = shot_in_end / 2;
    size_t sim_idx = (deliver_team == Team::k0 ? stone_idx : stone_idx + 8);
    if (!all_stones[sim_idx].has_value()) {
        all_stones[sim_idx] = ISimulator::StoneState(
            Vector2(),
            0.f,
            played_shot.ToVector2(),
            played_shot.angular_velocity
        );
        auto set_begin2 = std::chrono::steady_clock::now();
        simulator.SetStones(all_stones);
        AddNs(g_apply_move_profile.set_stones_ns, set_begin2, std::chrono::steady_clock::now());
        // std::cout << "Set stone index " << static_cast<int>(stone_idx) << " for team " << ((deliver_team == Team::k0) ? "0" : "1") << " in simulator." << std::endl;
    } else {
        // std::cout << "Error: Stone index " << static_cast<int>(stone_idx) << " for team " << ((deliver_team == Team::k0) ? "0" : "1") << " is already in play." << std::endl;
    }
    for (auto& stone : all_stones) {
        if (stone.has_value()) {
            // std::cout << "Initial stone at (" << stone->position.x << ", " << stone->position.y << ") with velocity (" << stone->translational_velocity.x << ", " << stone->translational_velocity.y << ") and angular velocity " << stone->angular_velocity << std::endl;
        }
    }

    // Simulate until all stones stop.
    // Keep simulator state sane by removing stones that left the valid sheet area,
    // matching the behavior of client_helpers::SimulateFull.
    auto step_begin = std::chrono::steady_clock::now();
    long long local_step_calls = 0;
    do {
        simulator.Step();
        ++local_step_calls;

        auto stepped_stones = simulator.GetStones();
        bool stone_removed = false;
        for (auto& stone : stepped_stones) {
            if (stone.has_value() && !IsValidMovingStone(stone.value(), game_setting.sheet_width)) {
                stone = std::nullopt;
                stone_removed = true;
                // std::cout << "A stone at (" << stone->position.x << ", " << stone->position.y << ") with velocity (" << stone->translational_velocity.x << ", " << stone->translational_velocity.y << ") left the sheet and is removed from simulation at " << local_step_calls << "th step" << std::endl;
            }
        }

        if (stone_removed) {
            simulator.SetStones(stepped_stones);
        }

        if (local_step_calls % 10 == 0) {
            for (auto& stone : stepped_stones) {
                if (stone.has_value()) {
                    // std::cout << "Stone at (" << stone->position.x << ", " << stone->position.y << ") with velocity (" << stone->translational_velocity.x << ", " << stone->translational_velocity.y << ") and angular velocity " << stone->angular_velocity << std::endl;
                }
            }
        }
    }
    while (!simulator.AreAllStonesStopped());
    AddNs(g_apply_move_profile.step_loop_ns, step_begin, std::chrono::steady_clock::now());
    g_apply_move_profile.step_calls.fetch_add(local_step_calls, std::memory_order_relaxed);

    // すべてのストーンが停止した後，プレイエリア外のストーンを除外
    {
        ISimulator::AllStones stones = simulator.GetStones();
        bool stone_removed = false;
        for (auto & stone : stones) {
            // std::cout << "Checking stone at (" << stone->position.x << ", " << stone->position.y << ")" << std::endl;
            if (stone && !IsValidStone(stone.value(), game_setting.sheet_width)) {
                stone = std::nullopt;
                stone_removed = true;
            }
        }
        if (stone_removed) {
            simulator.SetStones(stones);
        }
    }

    auto get_back_begin = std::chrono::steady_clock::now();
    StoneCoordinate simulated_stones = ConvertToStoneCoordinate(simulator.GetStones());
    AddNs(g_apply_move_profile.get_convert_back_ns, get_back_begin, std::chrono::steady_clock::now());

    auto validate_begin = std::chrono::steady_clock::now();
    game_state.stones = ValidateShot(game_rule, shot_in_end, deliver_team, before_stones, simulated_stones)
        ? simulated_stones
        : before_stones;

    // Advance shot/end
    int shots_per_end = (game_rule.type == GameRuleType::kMixedDoubles) ? 10 : 16;
    bool const is_end_finished = (static_cast<int>(shot_in_end) + 1) >= shots_per_end;
    if (is_end_finished) {
        FinalizeEnd(game_setting, game_state);
    } else {
        game_state.shot = static_cast<std::uint8_t>(shot_in_end + 1);
    }

    AddNs(g_apply_move_profile.validate_update_ns, validate_begin, std::chrono::steady_clock::now());
    AddNs(g_apply_move_profile.total_ns, total_begin, std::chrono::steady_clock::now());

    deliver_team = game_state.GetNextTeam();
    shot_in_end = game_state.shot;
    before_stones = game_state.stones;

    // std::cout << "Deliver team: " << ((deliver_team == Team::k0) ? "Team 0" : "Team 1") << ", Shot in end: " << static_cast<int>(shot_in_end) << std::endl;

    for (auto& stone : game_state.stones.GetAllStones()) {
        if (stone.has_value()) {
            // std::cout << "Final stone at (" << stone->position.x << ", " << stone->position.y << ")" << std::endl;
        }
    }

    return game_state;
}

} // namespace digitalcurling::client
