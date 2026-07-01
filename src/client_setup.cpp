// Copyright (c) 2022-2026 UEC Takeshi Ito Laboratory
// SPDX-License-Identifier: MIT

#include <iostream>
#include <stdexcept>
#include <string_view>
#include "digitalcurling/client/client_base.hpp"
#include "digitalcurling/client/client_factory.hpp"
#include "digitalcurling/client/protocol_models.hpp"
#include "digitalcurling/plugins/plugin_factory_creator.hpp"
#include "./engine/cnn_engine.hpp"

using StateUpdateEventData = digitalcurling::client::StateUpdateEventData;


// --- Registration ---
std::unique_ptr<digitalcurling::client::IFactoryCreator> CreateFactoryCreator() {
    return std::make_unique<digitalcurling::plugins::PluginFactoryCreator>();
}
std::unique_ptr<digitalcurling::client::IThinkingEngine> CreateThinkingEngine() {
    return std::make_unique<digitalcurling::client::CNNEngine>();
}
digitalcurling::client::ClientConnectSetting::Callback GetCallback() {
    return digitalcurling::client::ClientConnectSetting::Callback {
        nullptr, //digitalcurling::client::detail::OnConnected,
        nullptr, //digitalcurling::client::detail::OnLatestStateUpdate,
        nullptr, //digitalcurling::client::detail::OnStateUpdate,
        nullptr, //digitalcurling::client::detail::OnSSEProcessError
    };
}


/// --- Callbacks for connection and events ---
namespace digitalcurling::client::detail {

void OnConnected() {

}
void OnStateUpdate(StateUpdateEventData const& event_data) {

}
void OnLatestStateUpdate(StateUpdateEventData const& event_data) {

}
bool OnSSEProcessError(std::runtime_error const& e) {
    std::cout << "Do you want to continue the game? (y/n): ";
    char c;
    std::cin >> c;
    return c == 'y' || c == 'Y';
}

} // namespace digitalcurling::client::detail
