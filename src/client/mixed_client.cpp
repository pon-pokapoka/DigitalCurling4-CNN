#include "digitalcurling/client/mixed_client.hpp"

namespace digitalcurling::client {

std::unique_ptr<MixedClient> MixedClient::Create(
    std::string host,
    std::string id,
    MatchInfo const& match_info,
    std::unique_ptr<IMixedThinkingEngine> engine,
    std::unique_ptr<IFactoryCreator> factory_creator
) {
    std::vector<std::unique_ptr<players::IPlayerFactory>> players(4);
    std::vector<players::Gender> players_gender(4);
    for (int i = 0; i < 4; i++) {
        players[i] = factory_creator->CreatePlayerFactory(match_info.players[i]);
        players_gender[i] = players[i]->GetGender();
    }

    auto idxs = engine->OnInit(
        match_info.rule,
        match_info.setting,
        factory_creator->CreateSimulatorFactory(match_info.simulator),
        std::move(players)
    );
    if (idxs.size() != 4)
        throw std::runtime_error("MixedClient: Number of players after OnInit is not 4");

    auto last_gender = players::Gender::kUnknown;
    for (int i = 0; i < 4; i++) {
        auto g = players_gender[idxs[i]];
        if (last_gender == g) {
            throw std::runtime_error("MixedClient: Consecutive players must be of the opposite sex. (index: " + std::to_string(i) + ")");
        }
        last_gender = g;
    }
    return std::unique_ptr<MixedClient>(
        new MixedClient(std::move(host), std::move(id), match_info, std::move(engine), std::move(factory_creator), std::move(players), std::move(idxs))
    );
}

GameRuleType MixedClient::GetType() const {
    return GameRuleType::kMixed;
}

std::vector<std::uint8_t> MixedClient::GetPlayersIndex() const {
    return players_index_;
}

void MixedClient::OnGameStart(
    Team const& team,
    std::vector<std::pair<GameState, std::optional<moves::Shot>>> states
) {
    engine_->OnGameStart(team, std::move(states));
}

void MixedClient::OnNextEnd(StateUpdateEventData const& event_data) {
    engine_->OnNextEnd(event_data.game_state);
}
moves::Move MixedClient::OnMyTurn(StateUpdateEventData const& event_data) {
    if (players_index_.empty()) throw std::runtime_error("MixedClient: players_index_ is empty");
    std::size_t index = static_cast<std::size_t>(event_data.game_state.shot / 2) / players_index_.size();
    return engine_->OnMyTurn(
        players_[players_index_[index]], event_data.game_state, event_data.last_shot
    );
}
void MixedClient::OnOpponentTurn(StateUpdateEventData const& event_data) {
    engine_->OnOpponentTurn(event_data.game_state, event_data.last_shot);
}

void MixedClient::OnGameOver(StateUpdateEventData const& event_data) {
    engine_->OnGameOver(event_data.game_state);
}

} // namespace digitalcurling::client
