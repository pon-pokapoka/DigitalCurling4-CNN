// Copyright (c) 2026 Takehiro Yoshioka
// SPDX-License-Identifier: MIT

// Which model and win table this client loaded, so results can be traced to
// them: the client is its code, its model and its win table together.

#ifndef MODEL_IDENTITY_HPP
#define MODEL_IDENTITY_HPP

#include <filesystem>
#include <string>

namespace digitalcurling::client {

// Digits of a SHA-256 hex digest used as a short ID, matching the
// <name>-<first 8 hex>.pt / .csv naming of the model files.
constexpr std::size_t kShortHashLength = 8;

// Directory holding traced.pt and win_table.csv, found from the executable's
// path (<root>/build/bin/<exe> -> <root>/src/engine/model). Throws if the
// executable's path can't be determined.
std::filesystem::path ModelDir();

std::filesystem::path ModelPath();     // ModelDir() / "traced.pt"
std::filesystem::path WinTablePath();  // ModelDir() / "win_table.csv"

struct ModelIdentity {
    std::string model_file;   // file name after resolving symlinks, e.g. gen0-588f425c.pt
    std::string model_hash;   // full SHA-256 hex
    std::string table_file;
    std::string table_hash;

    // "<model hash prefix>-<table hash prefix>", e.g. 588f425c-648e2603
    std::string ShortId() const;
};

// Hashes ModelPath() and WinTablePath() on the first call and returns the same
// result afterwards. Throws if either file can't be read.
ModelIdentity const& LoadedModelIdentity();

// The team name sent to the server. Short by default, so external servers such
// as tournaments see the registered name. With `full`, it also carries the
// build ID and the model and table hashes, for internal servers:
// <name>-<build id>-<model hash prefix>-<table hash prefix>.
std::string ComposeTeamName(std::string const& name, bool full,
                            std::string const& build_id, ModelIdentity const& identity);

// True if the environment variable `name` is set to 1 or true.
bool EnvFlag(char const* name);

} // namespace digitalcurling::client

#endif // MODEL_IDENTITY_HPP
