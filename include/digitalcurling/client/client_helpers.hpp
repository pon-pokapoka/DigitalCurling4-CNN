// Copyright (c) 2022-2026 UEC Takeshi Ito Laboratory
// SPDX-License-Identifier: Unlicense

#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <optional>
#include <digitalcurling/game_rule.hpp>
#include <digitalcurling/game_setting.hpp>
#include <digitalcurling/game_state.hpp>
#include <digitalcurling/stone_index.hpp>
#include <digitalcurling/stone_coordinate.hpp>
#include <digitalcurling/moves/shot.hpp>
#include <digitalcurling/rules/i_additional_rule.hpp>
#include <digitalcurling/simulators/i_simulator.hpp>

namespace digitalcurling::client {

/// @brief ショットの情報
struct ShotInfo {
    /// @brief ショットを行うストーンのインデックス
    digitalcurling::StoneIndex index;

    /// @brief ショットのパラメータ
    digitalcurling::moves::Shot shot;
};

/// @brief Convert a remaining time from the server to milliseconds.
/// @param seconds Remaining time in seconds; negative once a team has run out of time
/// @return The remaining time, 0 if it is negative
inline std::chrono::milliseconds RemainingTimeFromSeconds(double seconds) {
    return std::chrono::milliseconds(static_cast<std::int64_t>(std::max(seconds, 0.0) * 1000));
}

/// @brief 盤面をシミュレータ用のストーン配列に変換する
/// @param stone_coordinate 変換元の盤面
/// @return シミュレータ用のストーン配列
inline simulators::ISimulator::AllStones ConvertToSimulatorStones(StoneCoordinate const& stone_coordinate) {
    auto const pre_shot_stones = stone_coordinate.GetAllStones();
    simulators::ISimulator::AllStones stones;
    std::transform(pre_shot_stones.begin(), pre_shot_stones.end(), stones.begin(),
        [](const std::optional<Stone>& s) {
            return !s.has_value() ? std::nullopt :
                std::make_optional<simulators::ISimulator::StoneState>(
                    {s->position, s->angle, {}, 0.f}
                );
        }
    );
    return stones;
}

/// @brief シミュレータから盤面を取得する
/// @param simulator シミュレーター
/// @return 盤面
inline StoneCoordinate GetStoneCoordinateFromSimulator(simulators::ISimulator* simulator) {
    std::array<std::optional<Stone>, 16> stones_array;
    auto stones = simulator->GetStones();
    for (int i = 0; i < 16; i++) stones_array[i] = std::move(stones[i]);
    return StoneCoordinate(stones_array);
}

/// @brief シミュレータ上のストーンが盤面上で有効かを判定する
/// @param stone ストーン
/// @param sheet_width シートの幅
/// @return ストーンが盤面上で有効なら `true`
inline bool IsVaildStone(Stone const& stone, float sheet_width) {
    return stone.position.x + Stone::kRadius < sheet_width / 2.f
        && stone.position.x - Stone::kRadius > -sheet_width / 2.f
        && stone.position.y - Stone::kRadius < coordinate::kBackLineY
        && stone.position.y - Stone::kRadius > coordinate::kBackBoardY;
}
/// @brief シミュレータをストーンが全て停止するまで進める
/// @param simulator シミュレーター
/// @param sheet_width シートの幅
inline void SimulateFull(simulators::ISimulator* simulator, float sheet_width) {
    do {
        simulator->Step();

        auto stones = simulator->GetStones();
        bool stone_removed = false;
        for (auto & stone : stones) {
            if (stone.has_value() && !IsVaildStone(stone.value(), sheet_width)) {
                stone = std::nullopt;
                stone_removed = true;
            }
        }

        if (stone_removed) simulator->SetStones(stones);
    }
    while (!simulator->AreAllStonesStopped());
}

} // namespace digitalcurling::client
