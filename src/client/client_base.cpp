// Copyright (c) 2022-2026 UEC Takeshi Ito Laboratory
// SPDX-License-Identifier: Unlicense

#ifndef DIGITALCURLING_CLIENT_NAME
    #error "DIGITALCURLING_CLIENT_NAME is not defined"
#endif
#ifndef DIGITALCURLING_CLIENT_VERSION_MAJOR
    #error "DIGITALCURLING_CLIENT_VERSION_MAJOR is not defined"
#endif

#include <algorithm>
#include <iostream>
#include <memory>
#include <thread>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <nlohmann/json.hpp>
#include "digitalcurling/client/client_base.hpp"
#include "digitalcurling/client/client_helpers.hpp"
#include "build_id.hpp"
#include "engine/model_identity.hpp"

using json = nlohmann::json;

namespace digitalcurling::client {

ClientBase::ClientBase(std::string host, std::string id, MatchInfo const& match_info)
  : host_(std::move(host)),
    game_id_(std::move(id)),
    team_(Team::kInvalid),
    rule_type_(match_info.rule.type),
    max_end_(match_info.setting.max_end),
    players_(match_info.players),
    sse_headers_(),
    http_client_(host_)
{
    http_client_.set_connection_timeout(10, 0);
    http_client_.set_read_timeout(10, 0);
}

std::string ClientBase::GetName() const
{
    return std::string(DIGITALCURLING_CLIENT_NAME);
}

Version ClientBase::GetVersion() const
{
    return Version(
        DIGITALCURLING_CLIENT_VERSION_MAJOR
    #ifdef DIGITALCURLING_CLIENT_VERSION_MINOR
        , DIGITALCURLING_CLIENT_VERSION_MINOR
    #  ifdef DIGITALCURLING_CLIENT_VERSION_PATCH
        , DIGITALCURLING_CLIENT_VERSION_PATCH
    #  endif
    #endif
    );
}

Version ClientBase::GetProtocolVersion() const {
    return {1, 0};
}

Team ClientBase::JoinGame(Team const& team, std::string const& auth_id, std::string const& auth_pw)
{
    http_client_.set_basic_auth(auth_id, auth_pw);

    // The short name by default, for external servers such as tournaments.
    // DC_TEAM_NAME_FULL=1 adds the build ID and the model and table hashes, so
    // an internal server's records show exactly which client played.
    bool const full_name = EnvFlag("DC_TEAM_NAME_FULL");
    std::string const team_name = ComposeTeamName(
        GetName(), full_name, DIGITALCURLING_CLIENT_BUILD_ID,
        full_name ? LoadedModelIdentity() : ModelIdentity{});
    std::cout << "Team name: " << team_name << std::endl;

    nlohmann::json pl_json = {
        { "use_default_config", true },
        { "team_name", team_name }
    };
    auto idxs = GetPlayersIndex();

    for (int i = 0; i < idxs.size(); i++) {
        auto j = players_.at(idxs[i]);
        nlohmann::json player_json = {
            { "max_velocity", j.at("max_speed").get<float>() },
            { "shot_std_dev", j.at("stddev_speed").get<float>() },
            { "angle_std_dev", j.at("stddev_angle").get<float>() },
            { "player_name", "player" + std::to_string(idxs[i]) },
        };
        pl_json["player" + std::to_string(i + 1)] = player_json;
    }

    const std::string team_path = "/store-team-config?match_id=" + game_id_
            + "&expected_match_team_name=" + ToString(team);

    auto result = http_client_.Post(team_path, pl_json.dump(), "application/json");
    if (!result) {
        throw std::runtime_error("Failed to join a game: " + httplib::to_string(result.error()));
    } else if (result->status != 200) {
        if (result->status == 401) {
            throw std::runtime_error("Failed to join a game: authentication is failed.");
        }
        std::string err = "Failed to get match information: return status code " + std::to_string(result->status);
        if (!result->body.empty()) err += " " + result->body;
        throw std::runtime_error(err);
    }

    if (result->body == "\"team0\"") {
        team_ = Team::k0;
    } else if (result->body == "\"team1\"") {
        team_ = Team::k1;
    } else {
        throw std::runtime_error("Failed to join a game: unknown team received: " + result->body);
    }

    players_.clear();
    auto auth_base64 = httplib::detail::base64_encode(auth_id + ":" + auth_pw);
    sse_headers_.emplace("Authorization", "Basic " + auth_base64);
    return team_;
};

void ClientBase::Connect(ClientConnectSetting const& setting) {
    if (team_ == Team::kInvalid) {
        throw std::runtime_error("ClientBase::Connect: team is not set. Call JoinGame() first.");
    }

    const std::string game_path = "/matches/" + game_id_ + "/stream";
    auto sse_http_client = httplib::Client(host_);
    auto sse_client = httplib::sse::SSEClient(sse_http_client, game_path, sse_headers_);
    sse_client.set_max_reconnect_attempts(setting.max_retry_count);
    sse_client.set_reconnect_interval(setting.retry_wait_time.count());

    // One slot per writer, so no lock is needed and a reconnect can't hide or
    // replace a processing error. SSE callbacks run on this thread, inside start().
    std::optional<std::runtime_error> connection_error;
    std::optional<std::runtime_error> processing_error; // read after the join below
    std::atomic<bool> is_sse_stopped = false;

    std::mutex queue_mutex;
    std::condition_variable cond_var;
    std::queue<std::pair<std::string, std::function<void()>>> event_queue;

    auto push_event = [&](std::string event_name, std::function<void()> handler) {
        std::lock_guard lock(queue_mutex);
        event_queue.push({std::move(event_name), std::move(handler)});
        cond_var.notify_one();
    };

    sse_client.on_open([&]() {
        connection_error = std::nullopt;
        push_event("on_connected", [&setting]() {
            if (setting.callback.on_connected)
                setting.callback.on_connected();
        });
    });
    sse_client.on_error([&](httplib::Error err) {
        connection_error = std::runtime_error("SSE connection error: " + httplib::to_string(err));
    });
    sse_client.on_event("latest_state_update", [&](const httplib::sse::SSEMessage &msg) {
        // std::cout << "Received latest_state_update event: " << msg.data << std::endl << std::endl;
        push_event("latest_state_update event", [this, &setting, &sse_client, msg]() {
            StateUpdateEventData event_data = ParseStateUpdateEventData(msg);
            if (setting.callback.on_latest_state_update)
                setting.callback.on_latest_state_update(event_data);
            OnReceiveLatestStateUpdateEvent(event_data);

            if (event_data.game_state.IsGameOver()) sse_client.stop();
        });
    });
    sse_client.on_event("state_update", [&](const httplib::sse::SSEMessage &msg) {
        // std::cout << "Received state_update event: " << msg.data << std::endl << std::endl;
        push_event("state_update event", [this, &setting, &sse_client, msg]() {
            StateUpdateEventData event_data = ParseStateUpdateEventData(msg);
            if (setting.callback.on_state_update)
                setting.callback.on_state_update(event_data);
            OnReceiveStateUpdateEvent(event_data);
        });
    });

    std::thread processing_thread = std::thread([&]() {
        while (true) {
            std::pair<std::string, std::function<void()>> event;
            {
                std::unique_lock lock(queue_mutex);
                cond_var.wait(lock, [&]{
                    return !event_queue.empty() || is_sse_stopped;
                });

                if (is_sse_stopped && event_queue.empty()) break;
                if (event_queue.empty()) continue;

                event = std::move(event_queue.front());
                event_queue.pop();
            }
            try {
                event.second();
            } catch (std::exception const& e) {
                auto err = std::runtime_error("Exception occurred while processing " + event.first + ": " + e.what());
                if (!setting.callback.on_event_process_error || !setting.callback.on_event_process_error(err)) {
                    if (!processing_error) processing_error = std::move(err);
                    sse_client.stop();
                }
            } catch (...) {
                if (!processing_error) processing_error = std::runtime_error("Unknown exception occurred in event handling thread.");
                sse_client.stop();
            }
        }
    });

    sse_client.start();
    {
        std::lock_guard lock(queue_mutex);
        is_sse_stopped = true;
    }
    cond_var.notify_all();
    processing_thread.join();

    // A processing error is the root cause: a connection error after it comes from stopping.
    if (processing_error.has_value()) throw std::move(processing_error.value());
    if (connection_error.has_value()) throw std::move(connection_error.value());
}

StateUpdateEventData ClientBase::ParseStateUpdateEventData(httplib::sse::SSEMessage const& message) {
    auto json = json::parse(message.data);
    auto total_shot_opt = json.at("total_shot_number").get<std::optional<int>>();

    Team next_team; int total_shot;
    if (total_shot_opt.has_value()) {
        total_shot = total_shot_opt.value();
        next_team = json.at("next_shot_team").get<Team>();
    } else if (rule_type_ == GameRuleType::kMixedDoubles) {
        total_shot = 0;
        next_team = Team::kInvalid;
    } else {
        throw std::runtime_error("Invalid event data: total_shot_number is required for non-mixed-doubles game mode.");
    }

    GameState state;
    state.end = std::min(json.at("end_number").get<std::uint8_t>(), max_end_);
    state.thinking_time_remaining = {
        RemainingTimeFromSeconds(json.at("first_team_remaining_time").get<double>()),
        RemainingTimeFromSeconds(json.at("second_team_remaining_time").get<double>())
    };

    auto j_scores = json.at("score").get<TeamValue<std::vector<std::uint8_t>>>();
    std::vector<std::optional<std::uint8_t>> scores_team0(max_end_ + 1), scores_team1(max_end_ + 1);
    int end_completed = 0;
    if (total_shot == 16) end_completed = 1;
    std::cout << "\nEnd Completed: " << end_completed << "\n\n";
    // A short array from the server leaves those ends unset instead of reading
    // past it. It isn't an error, which would stop the client mid-game, but the
    // engine's score would be wrong, so say so.
    auto const& j_scores0 = j_scores[Team::k0];
    auto const& j_scores1 = j_scores[Team::k1];
    int const ends_to_read = state.end + end_completed;
    if (static_cast<int>(std::min(j_scores0.size(), j_scores1.size())) < ends_to_read) {
        std::cerr << "[Warning] score arrays from the server are shorter than the " << ends_to_read
                  << " ends played: " << j_scores0.size() << " and " << j_scores1.size() << std::endl;
    }
    for (int e = 0; e < ends_to_read; e++) {
        if (e < static_cast<int>(j_scores0.size())) scores_team0[e] = j_scores0[e];
        if (e < static_cast<int>(j_scores1.size())) scores_team1[e] = j_scores1[e];
    }
    state.scores = {{ std::move(scores_team0), std::move(scores_team1) }};

    std::optional<moves::Shot> last_shot = std::nullopt;
    if (total_shot == 0) {
        state.shot = 0;
        last_shot = std::nullopt;

        if (next_team != Team::kInvalid) {
            state.hammer = current_hammer_ = GetOpponentTeam(next_team);
        } else {
            state.hammer = json.at("mix_doubles_settings").at("end_setup_team").get<Team>();
        }
    } else {
        state.shot = static_cast<std::uint8_t>(total_shot);
        state.hammer = current_hammer_;
    }

    auto j_last_move = json.at("last_move");
    if (j_last_move.is_null()) {
        last_shot = std::nullopt;
    } else {
        last_shot = moves::Shot(
            j_last_move.at("translational_velocity").get<float>(),
            j_last_move.at("angular_velocity").get<float>(),
            j_last_move.at("shot_angle").get<float>()
        );
    }

    if (total_shot != 0 || (rule_type_ == GameRuleType::kMixedDoubles && next_team != Team::kInvalid)) {
        std::array<std::array<std::optional<Stone>, 8>, 2> state_stones {};
        auto stones = json.at("stone_coordinate").at("data").get<TeamValue<std::vector<Vector2>>>();

        for (std::uint8_t t = 0; t < 2; t++) {
            auto team = static_cast<Team>(t);
            for (int i = 0; i < stones[team].size(); i++) {
                const auto& src_stone = stones[team][i];
                if (src_stone != Vector2 {0.f, 0.f}) {
                    state_stones[t][i] = Stone { src_stone, 0.f };
                }
            }
        }

        if (rule_type_ == GameRuleType::kMixedDoubles) {
            std::swap(state_stones[0][0], state_stones[0][5]);
            std::swap(state_stones[1][0], state_stones[1][5]);
        }
        state.stones = StoneCoordinate(state_stones);
    } else {
        state.stones = StoneCoordinate();
    }

    auto winner = json.at("winner_team").get<std::optional<Team>>();
    if (winner.has_value()) {
        GameResult::Reason reason;
        if (state.thinking_time_remaining[GetOpponentTeam(winner.value())] > std::chrono::milliseconds(0)) {
            reason = GameResult::Reason::kScore;
        } else {
            reason = GameResult::Reason::kTimeLimit;
        }
        state.game_result = { winner.value(), reason };
    }

    return StateUpdateEventData { total_shot, next_team, std::move(state), std::move(last_shot) };
}

void ClientBase::OnReceiveLatestStateUpdateEvent(StateUpdateEventData const& event_data) {
    if (event_data.game_state.IsGameOver()) {
        OnGameOver(event_data);
        return;
    }

    if (is_first_update_) {
        is_first_update_ = false;
        OnGameStart(team_, std::move(states_));
    }
    if (event_data.total_shot_number == 0) {
        OnNextEnd(event_data);
    }

    if (event_data.next_shot_team == team_) {
        auto move = OnMyTurn(event_data);
        if (std::holds_alternative<moves::Shot>(move)) {
            const std::string shot_path = "/shots?match_id=" + game_id_;
            auto shot = std::get<moves::Shot>(move);
            nlohmann::json shot_json = {
                { "translational_velocity", shot.translational_velocity },
                { "angular_velocity", -shot.angular_velocity },
                { "shot_angle", shot.release_angle }
            };

            auto result = http_client_.Post(shot_path, shot_json.dump(), "application/json");
            if (!result) {
                throw std::runtime_error("Failed to post shot: " + httplib::to_string(result.error()));
            } else if (result->status != 200) {
                std::string err = "Failed to post shot: return status code " + std::to_string(result->status);
                if (!result->body.empty()) err += " " + result->body;
                throw std::runtime_error(err);
            }
        } else if (std::holds_alternative<moves::Concede>(move)) {
            // Currently, there is no API to concede a game.
        } else {
            throw std::runtime_error("Unknown move type returned by OnMyTurn.");
        }
    } else {
        OnOpponentTurn(event_data);
    }
}
void ClientBase::OnReceiveStateUpdateEvent(StateUpdateEventData const& event_data) {
    if (!is_first_update_) return;
    states_.emplace_back(event_data.game_state, event_data.last_shot);
}

} // namespace digitalcurling::client
