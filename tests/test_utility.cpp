// Unit tests for src/engine/utility.hpp (coordinate/pixel/shot conversions,
// index mapping, policy filters and model input encoding).

#include <gtest/gtest.h>

#include <torch/script.h>

#include <cmath>
#include <set>

#include "engine/utility.hpp"

namespace {

using digitalcurling::client::createFilter;
using digitalcurling::client::GameStateToInput;
using digitalcurling::client::Id3d1d;
using digitalcurling::client::Id4d1d;
using digitalcurling::client::PixelInBounds;
using digitalcurling::client::PixelToShot;
using digitalcurling::client::PositionToPixel;

constexpr int kPolicyPerRotation = config::policy_weight * config::policy_width;
constexpr int kPolicySize = kPolicyPerRotation * config::policy_rotation;

// Size of one sheet pixel in metres.
double PixelMetres() { return 1.0 / (config::m_to_inch * config::dpi); }

int PixelId(int rotation, int velocity_row, int column) {
    return Id3d1d(rotation, velocity_row, column);
}

// ---------------------------------------------------------------------------
// PositionToPixel / PixelInBounds
// ---------------------------------------------------------------------------

TEST(PositionToPixel, HogLineCentreMapsToRowZeroCentreColumn) {
    auto px = PositionToPixel(dc::Vector2(0.f, dc::coordinate::kHogLineY));
    EXPECT_EQ(px.first, 0);
    EXPECT_EQ(px.second, config::width / 2);
}

TEST(PositionToPixel, TeeIsInsideTheImage) {
    auto px = PositionToPixel(dc::coordinate::kTee);
    EXPECT_EQ(px.second, config::width / 2);
    EXPECT_TRUE(PixelInBounds(px.first, px.second));
    EXPECT_GT(px.first, 0);
}

TEST(PositionToPixel, OnePixelStepPerPixelSize) {
    double const step = PixelMetres();
    auto base = PositionToPixel(dc::Vector2(0.f, static_cast<float>(dc::coordinate::kHogLineY + 4 * step)));
    auto further = PositionToPixel(dc::Vector2(0.f, static_cast<float>(dc::coordinate::kHogLineY + 5 * step)));
    EXPECT_EQ(further.first - base.first, 1);
    EXPECT_EQ(further.second, base.second);
}

TEST(PositionToPixel, PositiveXMovesToSmallerColumn) {
    double const step = PixelMetres();
    float const y = dc::coordinate::kTeeLineY;
    auto centre = PositionToPixel(dc::Vector2(0.f, y));
    auto right = PositionToPixel(dc::Vector2(static_cast<float>(2 * step), y));
    auto left = PositionToPixel(dc::Vector2(static_cast<float>(-2 * step), y));
    EXPECT_EQ(right.second, centre.second - 2);
    EXPECT_EQ(left.second, centre.second + 2);
    EXPECT_EQ(right.first, centre.first);
}

TEST(PositionToPixel, EveryValidStonePositionIsInBounds) {
    // Area where ApplyMove keeps stones: inside the side lines, past the hog
    // line and not beyond the back line (with the stone radius).
    dc::GameSetting setting;
    float const r = dc::Stone::kRadius;
    float const x_max = setting.sheet_width / 2.f - r;
    float const y_min = dc::coordinate::kHogLineY + r;
    float const y_max = dc::coordinate::kBackLineY + r;
    int const n = 40;
    for (int i = 0; i <= n; ++i) {
        for (int j = 0; j <= n; ++j) {
            float x = -x_max + 2 * x_max * i / n;
            float y = y_min + (y_max - y_min) * j / n;
            auto px = PositionToPixel(dc::Vector2(x, y));
            EXPECT_TRUE(PixelInBounds(px.first, px.second))
                << "(" << x << ", " << y << ") -> (" << px.first << ", " << px.second << ")";
        }
    }
}

TEST(PixelInBounds, Edges) {
    EXPECT_TRUE(PixelInBounds(0, 0));
    EXPECT_TRUE(PixelInBounds(config::height - 1, config::width - 1));
    EXPECT_FALSE(PixelInBounds(-1, 0));
    EXPECT_FALSE(PixelInBounds(0, -1));
    EXPECT_FALSE(PixelInBounds(config::height, 0));
    EXPECT_FALSE(PixelInBounds(0, config::width));
}

// ---------------------------------------------------------------------------
// Index mapping
// ---------------------------------------------------------------------------

TEST(Id4d1d, IsRowMajorBijection) {
    int const batch = 2;
    int const total = batch * config::nChannel * config::height * config::width;
    std::vector<bool> seen(total, false);
    int expected = 0;
    for (int i = 0; i < batch; ++i)
        for (int l = 0; l < config::nChannel; ++l)
            for (int n = 0; n < config::height; ++n)
                for (int m = 0; m < config::width; ++m) {
                    int id = Id4d1d(i, l, n, m);
                    ASSERT_EQ(id, expected++);  // matches a {N, C, H, W} contiguous tensor
                    seen[id] = true;
                }
    for (bool s : seen) EXPECT_TRUE(s);
}

TEST(Id3d1d, IsRowMajorBijection) {
    int expected = 0;
    for (int r = 0; r < config::policy_rotation; ++r)
        for (int v = 0; v < config::policy_weight; ++v)
            for (int c = 0; c < config::policy_width; ++c)
                ASSERT_EQ(Id3d1d(r, v, c), expected++);
    EXPECT_EQ(expected, kPolicySize);
}

// ---------------------------------------------------------------------------
// PixelToShot
// ---------------------------------------------------------------------------

TEST(PixelToShot, AllPolicyIdsGiveSaneShots) {
    for (int id = 0; id < kPolicySize; ++id) {
        auto shot = PixelToShot(id);
        EXPECT_TRUE(std::isfinite(shot.translational_velocity)) << id;
        EXPECT_GT(shot.translational_velocity, 0.f) << id;
        EXPECT_GT(shot.release_angle, 0.f) << id;
        EXPECT_LT(shot.release_angle, static_cast<float>(M_PI)) << id;
        EXPECT_GT(shot.ToVector2().y, 0.f) << id;  // always thrown down the sheet
    }
}

TEST(PixelToShot, RotationSelectsSpinDirection) {
    for (int v = 0; v < config::policy_weight; ++v) {
        for (int c = 0; c < config::policy_width; ++c) {
            EXPECT_FLOAT_EQ(PixelToShot(PixelId(0, v, c)).angular_velocity,
                            static_cast<float>(config::default_angular_velocity));
            EXPECT_FLOAT_EQ(PixelToShot(PixelId(1, v, c)).angular_velocity,
                            static_cast<float>(-config::default_angular_velocity));
        }
    }
}

TEST(PixelToShot, VelocityDependsOnlyOnRowAndIncreases) {
    float previous = 0.f;
    for (int v = 0; v < config::policy_weight; ++v) {
        float const speed = PixelToShot(PixelId(0, v, 0)).translational_velocity;
        EXPECT_GT(speed, previous) << "row " << v;
        previous = speed;
        for (int r = 0; r < config::policy_rotation; ++r)
            for (int c = 0; c < config::policy_width; ++c)
                EXPECT_FLOAT_EQ(PixelToShot(PixelId(r, v, c)).translational_velocity, speed);
    }
}

TEST(PixelToShot, CentreColumnIsStraightDownTheSheet) {
    auto shot = PixelToShot(PixelId(0, 0, config::policy_width / 2));
    EXPECT_NEAR(shot.release_angle, M_PI / 2, 1e-6);
}

TEST(PixelToShot, ReleaseAngleDecreasesWithColumnAndIsSymmetric) {
    int const centre = config::policy_width / 2;
    for (int c = 1; c < config::policy_width; ++c) {
        EXPECT_LT(PixelToShot(PixelId(0, 0, c)).release_angle,
                  PixelToShot(PixelId(0, 0, c - 1)).release_angle) << c;
    }
    for (int d = 1; centre + d < config::policy_width; ++d) {
        float a = PixelToShot(PixelId(0, 0, centre - d)).release_angle;
        float b = PixelToShot(PixelId(0, 0, centre + d)).release_angle;
        EXPECT_NEAR(a + b, M_PI, 1e-5) << d;
    }
}

TEST(PixelToShot, RotationsShareVelocityAndAngle) {
    for (int v = 0; v < config::policy_weight; ++v) {
        for (int c = 0; c < config::policy_width; ++c) {
            auto a = PixelToShot(PixelId(0, v, c));
            auto b = PixelToShot(PixelId(1, v, c));
            EXPECT_FLOAT_EQ(a.translational_velocity, b.translational_velocity);
            EXPECT_FLOAT_EQ(a.release_angle, b.release_angle);
        }
    }
}

// ---------------------------------------------------------------------------
// Policy filters
// ---------------------------------------------------------------------------

TEST(CreateFilterFlat, MirrorSymmetricBetweenRotations) {
    auto filt = createFilter();
    int allowed[2] = {0, 0};
    for (int v = 0; v < config::policy_weight; ++v) {
        for (int c = 0; c < config::policy_width; ++c) {
            EXPECT_EQ(filt[Id3d1d(0, v, c)], filt[Id3d1d(1, v, config::policy_width - 1 - c)])
                << "row " << v << " col " << c;
            allowed[0] += filt[Id3d1d(0, v, c)];
            allowed[1] += filt[Id3d1d(1, v, c)];
        }
    }
    EXPECT_GT(allowed[0], 0);
    EXPECT_GT(allowed[1], 0);
}

TEST(CreateFilterFromState, MirrorSymmetricAndNonEmpty) {
    dc::GameSetting setting;
    dc::GameState state(setting);
    auto filt = createFilter(state, setting);
    int allowed = 0;
    for (int v = 0; v < config::policy_weight; ++v) {
        for (int c = 0; c < config::policy_width; ++c) {
            EXPECT_EQ(filt[0][v][c], filt[1][v][config::policy_width - 1 - c]);
            allowed += filt[0][v][c] + filt[1][v][c];
        }
    }
    EXPECT_GT(allowed, 0);
}

TEST(CreateFilterFromState, LastShotFilterIsStricterSubset) {
    dc::GameSetting setting;
    dc::GameState state(setting);
    state.shot = 14;
    auto normal = createFilter(state, setting);
    state.shot = 15;
    auto last = createFilter(state, setting);

    int normal_count = 0, last_count = 0;
    for (int r = 0; r < config::policy_rotation; ++r) {
        for (int v = 0; v < config::policy_weight; ++v) {
            for (int c = 0; c < config::policy_width; ++c) {
                if (last[r][v][c]) EXPECT_TRUE(normal[r][v][c]);
                normal_count += normal[r][v][c];
                last_count += last[r][v][c];
            }
        }
    }
    EXPECT_GT(last_count, 0);
    EXPECT_LT(last_count, normal_count);
    // Only the slowest rows are removed on the last shot.
    for (int r = 0; r < config::policy_rotation; ++r)
        for (int c = 0; c < config::policy_width; ++c)
            EXPECT_EQ(last[r][config::policy_weight - 1][c], normal[r][config::policy_weight - 1][c]);
}

// ---------------------------------------------------------------------------
// GameStateToInput
// ---------------------------------------------------------------------------

dc::GameState MakeState(dc::GameSetting const& setting) {
    dc::GameState state(setting);
    state.hammer = dc::Team::k1;
    return state;
}

TEST(GameStateToInput, ShapeDtypeAndDevice) {
    dc::GameSetting setting;
    std::vector<dc::GameState> states(3, MakeState(setting));
    auto input = GameStateToInput(states, setting, torch::kCPU, torch::kFloat);

    ASSERT_EQ(input.inputs.size(), 1u);
    auto t = input.inputs[0].toTensor();
    EXPECT_EQ(t.sizes(), (std::vector<int64_t>{3, config::nChannel, config::height, config::width}));
    EXPECT_EQ(t.scalar_type(), torch::kFloat);
    EXPECT_TRUE(t.device().is_cpu());
    EXPECT_EQ(input.end.size(), 3u);
    EXPECT_EQ(input.score.size(), 3u);
}

TEST(GameStateToInput, StonesGoToTeamChannels) {
    dc::GameSetting setting;
    auto state = MakeState(setting);  // hammer = team1
    dc::Vector2 const hammer_pos = dc::coordinate::kTee;
    dc::Vector2 const other_pos(0.5f, dc::coordinate::kTeeLineY - 1.0f);
    state.stones.team1[0] = dc::Stone(hammer_pos, 0.f);
    state.stones.team0[0] = dc::Stone(other_pos, 0.f);
    state.shot = 2;

    auto t = GameStateToInput({state}, setting, torch::kCPU, torch::kFloat).inputs[0].toTensor();
    auto hp = PositionToPixel(hammer_pos);
    auto op = PositionToPixel(other_pos);
    ASSERT_NE(hp, op);

    // channel 0: team without hammer, channel 1: team with hammer
    EXPECT_FLOAT_EQ(t[0][0][op.first][op.second].item<float>(), 1.f);
    EXPECT_FLOAT_EQ(t[0][1][hp.first][hp.second].item<float>(), 1.f);
    EXPECT_FLOAT_EQ(t[0][0].sum().item<float>(), 1.f);
    EXPECT_FLOAT_EQ(t[0][1].sum().item<float>(), 1.f);
}

TEST(GameStateToInput, HammerSwapSwapsChannels) {
    dc::GameSetting setting;
    auto state = MakeState(setting);
    dc::Vector2 const pos = dc::coordinate::kTee;
    state.stones.team0[0] = dc::Stone(pos, 0.f);

    state.hammer = dc::Team::k1;
    auto a = GameStateToInput({state}, setting, torch::kCPU, torch::kFloat).inputs[0].toTensor();
    state.hammer = dc::Team::k0;
    auto b = GameStateToInput({state}, setting, torch::kCPU, torch::kFloat).inputs[0].toTensor();

    EXPECT_FLOAT_EQ(a[0][0].sum().item<float>(), 1.f);
    EXPECT_FLOAT_EQ(a[0][1].sum().item<float>(), 0.f);
    EXPECT_FLOAT_EQ(b[0][0].sum().item<float>(), 0.f);
    EXPECT_FLOAT_EQ(b[0][1].sum().item<float>(), 1.f);
}

TEST(GameStateToInput, ShotChannelsEncodeShotNumber) {
    dc::GameSetting setting;
    float const plane = static_cast<float>(config::height * config::width);
    for (int shot : {0, 5, 15}) {
        auto state = MakeState(setting);
        state.shot = static_cast<std::uint8_t>(shot);
        auto t = GameStateToInput({state}, setting, torch::kCPU, torch::kFloat).inputs[0].toTensor();
        for (int l = 2; l < config::nChannel; ++l) {
            float const expected = (l - 2 <= shot) ? plane : 0.f;
            EXPECT_FLOAT_EQ(t[0][l].sum().item<float>(), expected) << "shot " << shot << " channel " << l;
        }
    }
}

TEST(GameStateToInput, ScoreIsFromHammerPerspective) {
    dc::GameSetting setting;
    auto state = MakeState(setting);
    state.scores.team0[0] = 2;
    state.scores.team1[0] = 0;
    state.scores.team0[1] = 0;
    state.scores.team1[1] = 1;
    state.end = 2;

    state.hammer = dc::Team::k1;
    auto a = GameStateToInput({state}, setting, torch::kCPU, torch::kFloat);
    EXPECT_EQ(a.score[0], -1);
    EXPECT_EQ(a.end[0], 2);

    state.hammer = dc::Team::k0;
    auto b = GameStateToInput({state}, setting, torch::kCPU, torch::kFloat);
    EXPECT_EQ(b.score[0], 1);
}

TEST(GameStateToInput, ExtraEndIsEncodedAsLastRegularEnd) {
    dc::GameSetting setting;
    auto state = MakeState(setting);
    state.end = setting.max_end;
    auto input = GameStateToInput({state}, setting, torch::kCPU, torch::kFloat);
    EXPECT_EQ(input.end[0], setting.max_end - 1);
}

TEST(GameStateToInput, FinishedGameIsAllZeros) {
    dc::GameSetting setting;
    auto state = MakeState(setting);
    state.stones.team0[0] = dc::Stone(dc::coordinate::kTee, 0.f);
    state.scores.team0[0] = 3;
    state.scores.team1[0] = 0;
    state.end = 1;
    state.shot = 4;
    state.game_result = dc::GameResult{dc::Team::k0, dc::GameResult::Reason::kConcede};

    auto input = GameStateToInput({state}, setting, torch::kCPU, torch::kFloat);
    EXPECT_FLOAT_EQ(input.inputs[0].toTensor().abs().sum().item<float>(), 0.f);
    EXPECT_EQ(input.end[0], 0);
    EXPECT_EQ(input.score[0], 0);
}

TEST(GameStateToInput, BatchEntriesAreIndependent) {
    dc::GameSetting setting;
    auto empty = MakeState(setting);
    auto with_stone = MakeState(setting);
    with_stone.stones.team1[3] = dc::Stone(dc::coordinate::kTee, 0.f);
    with_stone.shot = 7;

    auto t = GameStateToInput({empty, with_stone}, setting, torch::kCPU, torch::kFloat).inputs[0].toTensor();
    auto single = GameStateToInput({with_stone}, setting, torch::kCPU, torch::kFloat).inputs[0].toTensor();
    EXPECT_FLOAT_EQ(t[0][1].sum().item<float>(), 0.f);
    EXPECT_TRUE(torch::equal(t[1], single[0]));
}

TEST(ModelInput, ToCpuPreservesContents) {
    dc::GameSetting setting;
    auto state = MakeState(setting);
    state.end = 3;
    auto input = GameStateToInput({state}, setting, torch::kCPU, torch::kFloat);
    auto moved = input.to(torch::kCPU);
    EXPECT_TRUE(torch::equal(moved.inputs[0].toTensor(), input.inputs[0].toTensor()));
    EXPECT_EQ(moved.end, input.end);
    EXPECT_EQ(moved.score, input.score);
}

}  // namespace
