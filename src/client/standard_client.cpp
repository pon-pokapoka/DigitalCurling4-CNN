#include <algorithm>
#include "digitalcurling/client/standard_client.hpp"

namespace digitalcurling::client {

std::unique_ptr<StandardClient> StandardClient::Create(
    std::string host,
    std::string id,
    MatchInfo const& match_info,
    std::unique_ptr<IStandardThinkingEngine> engine,
    std::unique_ptr<IFactoryCreator> factory_creator
) {
    std::vector<std::unique_ptr<players::IPlayerFactory>> players(4);
    for (int i = 0; i < 4; i++) {
        players[i] = factory_creator->CreatePlayerFactory(match_info.players[i]);
    }

    nlohmann::json j;
    j["type"] = "fcv1light";
    j["seconds_per_frame"] = 0.001;

    auto idxs = engine->OnInit(
        match_info.rule,
        match_info.setting,
        factory_creator->CreateSimulatorFactory(j),
        std::move(players)
    );
    if (idxs.size() != 4)
        throw std::runtime_error("StandardClient: Number of players after OnInit is not 4");

    return std::unique_ptr<StandardClient>(
        new StandardClient(std::move(host), std::move(id), match_info, std::move(engine), std::move(factory_creator), std::move(players), std::move(idxs))
    );
}

GameRuleType StandardClient::GetType() const {
    return GameRuleType::kStandard;
}

std::vector<std::uint8_t> StandardClient::GetPlayersIndex() const {
    return players_index_;
}

void StandardClient::OnGameStart(
    Team const& team,
    std::vector<std::pair<GameState, std::optional<moves::Shot>>> states
) {
    engine_->OnGameStart(team, std::move(states));
}

void StandardClient::OnNextEnd(StateUpdateEventData const& event_data) {
    engine_->OnNextEnd(event_data.game_state);
}
moves::Move StandardClient::OnMyTurn(StateUpdateEventData const& event_data) {
    if (players_index_.empty()) throw std::runtime_error("StandardClient: players_index_ is empty");
    std::size_t index = static_cast<std::size_t>(event_data.game_state.shot / 2) / players_index_.size();
    return engine_->OnMyTurn(
        players_[players_index_[index]], event_data.game_state, event_data.last_shot
    );
}
void StandardClient::OnOpponentTurn(StateUpdateEventData const& event_data) {
    engine_->OnOpponentTurn(event_data.game_state, event_data.last_shot);
}

void StandardClient::OnGameOver(StateUpdateEventData const& event_data) {
    engine_->OnGameOver(event_data.game_state);
}

} // namespace digitalcurling::client
