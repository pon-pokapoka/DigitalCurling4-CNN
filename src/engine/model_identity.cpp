// Copyright (c) 2026 Takehiro Yoshioka
// SPDX-License-Identifier: MIT

#include "model_identity.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <stdexcept>

#include "executablepath.hpp"
#include "sha256.hpp"

namespace digitalcurling::client {

std::filesystem::path ModelDir()
{
    std::string const exec_path = getExecutablePath();
    if (exec_path.empty()) {
        throw std::runtime_error("cannot determine the executable path, needed to find the model files");
    }
    return std::filesystem::path(exec_path).parent_path().parent_path().parent_path() / "src" / "engine" / "model";
}

std::filesystem::path ModelPath() { return ModelDir() / "traced.pt"; }
std::filesystem::path WinTablePath() { return ModelDir() / "win_table.csv"; }

std::string ModelIdentity::ShortId() const
{
    return model_hash.substr(0, kShortHashLength) + "-" + table_hash.substr(0, kShortHashLength);
}

namespace {

std::string ResolvedFileName(std::filesystem::path const& path)
{
    std::error_code ec;
    auto const resolved = std::filesystem::canonical(path, ec);
    return (ec ? path : resolved).filename().string();
}

} // namespace

ModelIdentity const& LoadedModelIdentity()
{
    static ModelIdentity const identity = [] {
        ModelIdentity id;
        auto const model = ModelPath();
        auto const table = WinTablePath();
        id.model_file = ResolvedFileName(model);
        id.model_hash = FileSha256(model.string());
        id.table_file = ResolvedFileName(table);
        id.table_hash = FileSha256(table.string());
        return id;
    }();
    return identity;
}

std::string ComposeTeamName(std::string const& name, bool full,
                            std::string const& build_id, ModelIdentity const& identity)
{
    if (!full) return name;
    return name + "-" + build_id + "-" + identity.ShortId();
}

bool EnvFlag(char const* name)
{
    char const* value = std::getenv(name);
    if (value == nullptr) return false;
    std::string v(value);
    std::transform(v.begin(), v.end(), v.begin(), [](unsigned char c) { return std::tolower(c); });
    return v == "1" || v == "true";
}

} // namespace digitalcurling::client
