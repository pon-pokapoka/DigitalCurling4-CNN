// Copyright (c) 2022-2026 UEC Takeshi Ito Laboratory
// SPDX-License-Identifier: Unlicense

#include <memory>
#include <iostream>
#include <string>
#include <string_view>
#include <CLI/CLI.hpp>
#include "digitalcurling/client/client_factory.hpp"
#include "digitalcurling/client/client_base.hpp"
#include "digitalcurling/client/i_thinking_engine.hpp"
#include "build_id.hpp"

#ifdef DIGITALCURLING_CLIENT_USE_LOADER
    #include <digitalcurling/plugins/plugin_manager.hpp>
#endif

using namespace digitalcurling::client;

std::unique_ptr<IFactoryCreator> CreateFactoryCreator();
std::unique_ptr<IThinkingEngine> CreateThinkingEngine();
ClientConnectSetting::Callback GetCallback();

// --- Main function ---
constexpr std::string_view CLIENT_NAME = DIGITALCURLING_CLIENT_NAME;
constexpr std::string_view PROGRESS_HEADER = "Game in progress...";

int main(int argc, char const* argv[])
{
    /* Command line arguments */
    const std::string app_text = "Digital Curling Client: " + std::string(CLIENT_NAME)
        + " (build " + DIGITALCURLING_CLIENT_BUILD_ID + ")";

    CLI::App app{app_text};
    app.set_version_flag("-v,--version",
        app_text + "\n"
        "Library Version: " + digitalcurling::LibraryVersion.ToString()
    );

    bool is_debug, is_enable_console;
    app.add_flag("-d,--debug", is_debug, "Enable debug mode")->default_val(false)->force_callback();
    app.add_flag("-c,--console", is_enable_console, "Enable console input")->default_val(false)->force_callback();

    std::string host, id, auth_id, auth_pw;
    int team_idx;
    // Each option can also come from an environment variable; a command-line flag
    // wins. DC_AUTH_PW keeps the password out of the process's command line.
    app.add_option("--host", host, "The URL of the server to connect to")->default_str("")->envname("DC_HOST");
    app.add_option("--id", id, "The game/competition ID")->default_str("")->envname("DC_MATCH_ID");
    app.add_option("--team", team_idx, "The team index (0 or 1)")->default_val(0)->check(CLI::Range(0,1))->force_callback()->envname("DC_TEAM");
    app.add_option("--auth-id", auth_id, "The authentication ID")->default_str("user")->force_callback()->envname("DC_AUTH_ID");
    app.add_option("--auth-pw", auth_pw, "The authentication password")->default_str("password")->force_callback()->envname("DC_AUTH_PW");

#ifdef DIGITALCURLING_CLIENT_USE_LOADER
    std::string plugin_dir;
    app.add_option("--plugin-dir", plugin_dir, "The directory to load plugins from")->check(CLI::ExistingDirectory);
#endif

    CLI11_PARSE(app, argc, argv);

    if (host.empty() || id.empty()) {
        if (is_enable_console) {
            if (host.empty()) {
                id = "";
                std::cout << "Enter host: ";
                std::getline(std::cin, host);
            }
            if (id.empty()) {
                std::cout << "Enter ID: ";
                std::getline(std::cin, id);
            }
        } else {
            std::cerr << "[Error] Host and ID must be specified." << std::endl;
            std::cerr << app.help() << std::endl;
            return 1;
        }
    }

    std::cout << "\n" << app_text << "\n" << std::endl;
    if (is_debug) {
        std::cout << "[Debug] Debug mode enabled." << std::endl;
    }

#ifdef DIGITALCURLING_CLIENT_USE_LOADER
    /* Load plugins */
    std::cout << "Loading plugins ... ";

    std::string player_plugins = "", simulator_plugins = "";
    int loaded = 0;

    // check static loaded plugins
    auto manager = &digitalcurling::plugins::PluginManager::GetInstance();
    auto ids = manager->GetLoadedPlugins();
    for (const auto& id : ids) {
        if (id.type == digitalcurling::plugins::PluginType::player) {
            if (!player_plugins.empty()) player_plugins += ", ";
            player_plugins += id.name + "(static)";
        } else if (id.type == digitalcurling::plugins::PluginType::simulator) {
            if (!simulator_plugins.empty()) simulator_plugins += ", ";
            simulator_plugins += id.name + "(static)";
        }
        loaded++;
    }

    // load file from plugin_dir
    if (app.count("--plugin-dir") > 0 ||
        (std::filesystem::exists((plugin_dir = "./plugins")) && std::filesystem::is_directory(plugin_dir))) {
        for (const auto& entry : std::filesystem::directory_iterator(plugin_dir)) {
            if (entry.is_regular_file()) {
                try {
                    auto result = manager->LoadPlugin(entry.path(), false);
                    if (result.has_value()) {
                        if (result->type == digitalcurling::plugins::PluginType::player) {
                            if (!player_plugins.empty()) player_plugins += ", ";
                            player_plugins += result->name;
                        } else if (result->type == digitalcurling::plugins::PluginType::simulator) {
                            if (!simulator_plugins.empty()) simulator_plugins += ", ";
                            simulator_plugins += result->name;
                        }
                        loaded++;
                    }
                } catch (const std::exception& e) {
                    std::cerr << "[Error] Failed to load plugin from " << entry.path() << ": " << e.what() << std::endl;
                }
            }
        }
    }

    if (loaded == 0) {
        std::cout << "ERROR" << std::endl;
        std::cerr << "[Error] No plugins loaded. The client cannot function properly." << std::endl;
        return 1;
    }

    std::cout << "OK" << std::endl;
    std::cout << "[Plugins Info]\n"
        << "  Player Plugins: " << (player_plugins.empty() ? "None" : player_plugins) << "\n"
        << "  Simulator Plugins: " << (simulator_plugins.empty() ? "None" : simulator_plugins) << "\n"
        << std::endl;
#endif

    std::unique_ptr<ClientBase> client;
    try {
        std::cout << "Creating client ... ";
        std::unique_ptr<IFactoryCreator> factory_creator = CreateFactoryCreator();
        std::unique_ptr<IThinkingEngine> engine = CreateThinkingEngine();
        client = ClientFactory::CreateClient(host, id, std::move(engine), std::move(factory_creator));
        std::cout << "OK" << std::endl;

        std::cout << "Joining game ... ";
        auto team = client->JoinGame(static_cast<digitalcurling::Team>(team_idx), auth_id, auth_pw);
        std::cout << "OK\n\n[Game Info]\n"
            << "  Host: " << host << "\n"
            << "  Game ID: " << id << "\n"
            << "  Team: " << digitalcurling::ToString(team) << "\n"
            << std::endl;

        std::cout << "Connecting to server ... ";

        auto update_log = [](StateUpdateEventData const& event_data) {
            printf(
                "\033[1A\033[%zdG\033[K (end %d, shot %d)",
                PROGRESS_HEADER.size() + 1,
                event_data.game_state.end + 1,
                event_data.total_shot_number
            );
            std::cout << std::endl;
        };

        ClientConnectSetting setting;
        auto user_callback = GetCallback();
        setting.callback = {
            [&user_callback]() { // on_connected
                std::cout << "OK\n" << PROGRESS_HEADER << " (end 1, shot 0)" << std::endl;
                if (user_callback.on_connected) user_callback.on_connected();
            },
            [&user_callback, &update_log](StateUpdateEventData const& event_data) { // on_latest_state_update
                update_log(event_data);
                if (user_callback.on_latest_state_update) user_callback.on_latest_state_update(event_data);
            },
            [&user_callback, &update_log](StateUpdateEventData const& event_data) { // on_state_update
                update_log(event_data);
                if (user_callback.on_state_update) user_callback.on_state_update(event_data);
            },
            [is_debug, &user_callback](std::runtime_error const& e) { // on_event_process_error
                std::cout << "[Error] " << e.what() << std::endl;
                if (user_callback.on_event_process_error && user_callback.on_event_process_error(e)) {
                    std::cout << PROGRESS_HEADER << std::endl;
                    return true;
                }
                return false;
            }
        };

        client->Connect(setting);
    } catch (const std::exception& e) {
        std::cerr << "[Error] " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
