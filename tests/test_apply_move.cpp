// Unit tests for src/engine/apply_move.cpp (shot application, end scoring and
// game result). Uses a deterministic fake simulator/player so no physics
// plugin, GPU or model file is needed.

#include <gtest/gtest.h>

#include <cstdlib>
#include <optional>

#include "engine/apply_move.hpp"

namespace {

namespace dc = digitalcurling;
using dc::client::ApplyMove;
using dc::simulators::ISimulator;

// Fake simulator: every stone that is moving when Step() is called is moved
// straight to `landing` (or left at the origin, i.e. short of the hog line,
// when `landing` is empty) and stopped.
class FakeSimulator : public ISimulator {
public:
    std::optional<dc::Vector2> landing;
    int step_calls = 0;
    int set_stones_calls = 0;

    void SetStones(AllStones const& stones) override {
        stones_ = stones;
        ++set_stones_calls;
    }

    void Step() override {
        ++step_calls;
        for (auto& stone : stones_) {
            if (!stone || IsStopped(*stone)) continue;
            stone->position = landing.value_or(dc::Vector2(0.f, 0.f));
            stone->translational_velocity = dc::Vector2(0.f, 0.f);
            stone->angular_velocity = 0.f;
        }
    }

    AllStones const& GetStones() const override { return stones_; }
    std::vector<Collision> const& GetCollisions() const override { return collisions_; }

    bool AreAllStonesStopped() const override {
        for (auto const& stone : stones_) {
            if (stone && !IsStopped(*stone)) return false;
        }
        return true;
    }

    float GetSecondsPerFrame() const override { return 0.001f; }
    const char* GetId() const noexcept override { return "fake_simulator"; }
    dc::simulators::ISimulatorFactory const& GetFactory() const override { std::abort(); }
    std::unique_ptr<dc::simulators::ISimulatorStorage> CreateStorage() const override { return nullptr; }
    void Save(dc::simulators::ISimulatorStorage&) const override {}
    void Load(dc::simulators::ISimulatorStorage const&) override {}

private:
    static bool IsStopped(StoneState const& s) {
        return s.translational_velocity.x == 0.f && s.translational_velocity.y == 0.f;
    }

    AllStones stones_{};
    std::vector<Collision> collisions_;
};

// Fake player: plays the requested shot exactly and records the calls.
class FakePlayer : public dc::players::IPlayer {
public:
    int play_calls = 0;
    dc::moves::Shot last_shot;

    dc::moves::Shot Play(dc::moves::Shot const& shot) override {
        ++play_calls;
        last_shot = shot;
        return shot;
    }
    dc::players::Gender GetGender() const override { return dc::players::Gender::kUnknown; }
    const char* GetId() const noexcept override { return "fake_player"; }
    dc::players::IPlayerFactory const& GetFactory() const override { std::abort(); }
    std::unique_ptr<dc::players::IPlayerStorage> CreateStorage() const override { return nullptr; }
    void Save(dc::players::IPlayerStorage&) const override {}
    void Load(dc::players::IPlayerStorage const&) override {}
};

dc::Vector2 const kTee = dc::coordinate::kTee;

dc::Vector2 InHouse(float distance_from_tee) {
    return dc::Vector2(0.f, dc::coordinate::kTeeLineY - distance_from_tee);
}

// In play (past the hog line) but outside the house.
dc::Vector2 Guard() {
    return dc::Vector2(0.f, dc::coordinate::kHogLineY + 1.0f);
}

class ApplyMoveTest : public ::testing::Test {
protected:
    ApplyMoveTest() : state_(setting_) {
        rule_.type = dc::GameRuleType::kStandard;
        state_.hammer = dc::Team::k1;
    }

    dc::GameState Apply(dc::moves::Move const& move) {
        return ApplyMove(rule_, setting_, sim_, player_, state_, move, std::chrono::milliseconds(0));
    }

    dc::GameState Throw(std::optional<dc::Vector2> landing) {
        sim_.landing = landing;
        return Apply(dc::moves::Shot(2.4f, 1.f, static_cast<float>(M_PI / 2)));
    }

    // Puts the state at the last shot of `end` with prior ends scored 0-0.
    void GoToLastShot(int end) {
        state_.end = static_cast<std::uint8_t>(end);
        state_.shot = static_cast<std::uint8_t>(15);
        int const slots = static_cast<int>(state_.scores.team0.size());
        for (int e = 0; e < end && e < slots; ++e) {
            state_.scores.team0[e] = 0;
            state_.scores.team1[e] = 0;
        }
    }

    dc::GameRule rule_;
    dc::GameSetting setting_;
    dc::GameState state_;
    FakeSimulator sim_;
    FakePlayer player_;
};

// ---------------------------------------------------------------------------
// Shot application
// ---------------------------------------------------------------------------

TEST_F(ApplyMoveTest, FirstShotIsDeliveredByTeamWithoutHammer) {
    auto result = Throw(kTee);

    EXPECT_EQ(player_.play_calls, 1);
    EXPECT_GT(sim_.step_calls, 0);
    ASSERT_TRUE(state_.stones.team0[0].has_value());
    EXPECT_FLOAT_EQ(state_.stones.team0[0]->position.y, kTee.y);
    EXPECT_EQ(state_.stones.GetActiveStones().size(), 1u);
    EXPECT_EQ(state_.shot, 1);
    EXPECT_EQ(state_.end, 0);
    EXPECT_FALSE(state_.IsGameOver());
    // The returned state is the updated state.
    EXPECT_EQ(result.shot, state_.shot);
    EXPECT_EQ(result.stones.GetActiveStones().size(), 1u);
}

TEST_F(ApplyMoveTest, StoneSlotFollowsShotNumberAndTeam) {
    state_.shot = 5;  // odd shot: hammer team, its 3rd stone (index 2)
    Throw(InHouse(1.0f));
    EXPECT_TRUE(state_.stones.team1[2].has_value());
    EXPECT_EQ(state_.shot, 6);

    Throw(InHouse(0.5f));  // shot 6: non-hammer team, index 3
    EXPECT_TRUE(state_.stones.team0[3].has_value());
    EXPECT_EQ(state_.shot, 7);
}

TEST_F(ApplyMoveTest, PlayerReceivesRequestedShot) {
    dc::moves::Shot shot(2.3f, -1.f, 1.5f);
    sim_.landing = kTee;
    Apply(shot);
    EXPECT_FLOAT_EQ(player_.last_shot.translational_velocity, shot.translational_velocity);
    EXPECT_FLOAT_EQ(player_.last_shot.angular_velocity, shot.angular_velocity);
    EXPECT_FLOAT_EQ(player_.last_shot.release_angle, shot.release_angle);
}

TEST_F(ApplyMoveTest, StoneShortOfHogLineIsRemoved) {
    Throw(std::nullopt);
    EXPECT_TRUE(state_.stones.GetActiveStones().empty());
    EXPECT_EQ(state_.shot, 1);
}

TEST_F(ApplyMoveTest, StoneBeyondBackLineIsRemoved) {
    Throw(dc::Vector2(0.f, dc::coordinate::kBackLineY + 1.0f));
    EXPECT_TRUE(state_.stones.GetActiveStones().empty());
}

TEST_F(ApplyMoveTest, StoneOverSideLineIsRemoved) {
    Throw(dc::Vector2(setting_.sheet_width / 2.f, dc::coordinate::kTeeLineY));
    EXPECT_TRUE(state_.stones.GetActiveStones().empty());
}

TEST_F(ApplyMoveTest, ExistingStonesArePreserved) {
    state_.stones.team1[0] = dc::Stone(Guard(), 0.f);
    state_.shot = 2;
    Throw(kTee);
    EXPECT_TRUE(state_.stones.team1[0].has_value());
    EXPECT_TRUE(state_.stones.team0[1].has_value());
    EXPECT_EQ(state_.stones.GetActiveStones().size(), 2u);
}

TEST_F(ApplyMoveTest, ConcedeMakesOpponentOfNextTeamWin) {
    state_.shot = 3;  // next team: hammer (team1)
    auto result = Apply(dc::moves::Concede());

    ASSERT_TRUE(state_.IsGameOver());
    EXPECT_EQ(state_.game_result->winner, dc::Team::k0);
    EXPECT_EQ(state_.game_result->reason, dc::GameResult::Reason::kConcede);
    EXPECT_EQ(state_.shot, 3);
    EXPECT_EQ(player_.play_calls, 0);
    EXPECT_TRUE(result.IsGameOver());
}

TEST_F(ApplyMoveTest, FinishedGameIsUnchanged) {
    state_.game_result = dc::GameResult{dc::Team::k0, dc::GameResult::Reason::kScore};
    state_.shot = 4;
    auto result = Throw(kTee);

    EXPECT_EQ(player_.play_calls, 0);
    EXPECT_EQ(sim_.step_calls, 0);
    EXPECT_EQ(state_.shot, 4);
    EXPECT_TRUE(state_.stones.GetActiveStones().empty());
    EXPECT_EQ(result.game_result->winner, dc::Team::k0);
}

// ---------------------------------------------------------------------------
// End scoring
// ---------------------------------------------------------------------------

TEST_F(ApplyMoveTest, HammerTeamScoresAllStonesCloserThanOpponent) {
    GoToLastShot(0);
    state_.stones.team1[0] = dc::Stone(InHouse(0.5f), 0.f);
    state_.stones.team0[0] = dc::Stone(InHouse(1.0f), 0.f);
    state_.stones.team1[1] = dc::Stone(InHouse(1.5f), 0.f);  // behind opponent: no point
    Throw(kTee);  // last stone (team1) to the button

    EXPECT_EQ(state_.scores.team1[0], std::optional<std::uint8_t>(2));
    EXPECT_EQ(state_.scores.team0[0], std::optional<std::uint8_t>(0));
    EXPECT_EQ(state_.hammer, dc::Team::k0);  // scoring team gives up the hammer
    EXPECT_EQ(state_.end, 1);
    EXPECT_EQ(state_.shot, 0);
    EXPECT_TRUE(state_.stones.GetActiveStones().empty());
    EXPECT_FALSE(state_.IsGameOver());
}

TEST_F(ApplyMoveTest, StealKeepsHammerWithOpponent) {
    GoToLastShot(0);
    state_.stones.team0[0] = dc::Stone(InHouse(0.2f), 0.f);
    state_.stones.team0[1] = dc::Stone(InHouse(0.4f), 0.f);
    state_.stones.team0[2] = dc::Stone(InHouse(0.6f), 0.f);
    Throw(InHouse(1.0f));

    EXPECT_EQ(state_.scores.team0[0], std::optional<std::uint8_t>(3));
    EXPECT_EQ(state_.scores.team1[0], std::optional<std::uint8_t>(0));
    EXPECT_EQ(state_.hammer, dc::Team::k1);
}

TEST_F(ApplyMoveTest, BlankEndKeepsHammer) {
    GoToLastShot(0);
    state_.stones.team0[0] = dc::Stone(Guard(), 0.f);  // in play, outside the house
    Throw(std::nullopt);

    EXPECT_EQ(state_.scores.team0[0], std::optional<std::uint8_t>(0));
    EXPECT_EQ(state_.scores.team1[0], std::optional<std::uint8_t>(0));
    EXPECT_EQ(state_.hammer, dc::Team::k1);
    EXPECT_EQ(state_.end, 1);
    EXPECT_TRUE(state_.stones.GetActiveStones().empty());
}

TEST_F(ApplyMoveTest, StoneTouchingHouseEdgeCounts) {
    GoToLastShot(0);
    float const just_inside = dc::coordinate::kHouseRadius + dc::Stone::kRadius - 0.01f;
    Throw(InHouse(just_inside));
    EXPECT_EQ(state_.scores.team1[0], std::optional<std::uint8_t>(1));
}

TEST_F(ApplyMoveTest, MixedDoublesEndHasTenShots) {
    rule_.type = dc::GameRuleType::kMixedDoubles;
    state_.shot = 8;
    Throw(InHouse(1.0f));
    EXPECT_EQ(state_.end, 0);
    EXPECT_EQ(state_.shot, 9);

    Throw(kTee);  // 10th shot ends the end
    EXPECT_EQ(state_.end, 1);
    EXPECT_EQ(state_.shot, 0);
    EXPECT_EQ(state_.scores.team1[0], std::optional<std::uint8_t>(1));
}

// ---------------------------------------------------------------------------
// Game result
// ---------------------------------------------------------------------------

TEST_F(ApplyMoveTest, GameContinuesBeforeLastEnd) {
    GoToLastShot(setting_.max_end - 2);
    state_.scores.team0[0] = 5;
    Throw(std::nullopt);
    EXPECT_FALSE(state_.IsGameOver());
    EXPECT_EQ(state_.end, setting_.max_end - 1);
}

TEST_F(ApplyMoveTest, LastEndDecidesWinnerByTotalScore) {
    GoToLastShot(setting_.max_end - 1);
    state_.scores.team0[0] = 1;
    state_.stones.team1[0] = dc::Stone(InHouse(0.3f), 0.f);
    Throw(kTee);  // team1 scores 2 in the last end: 2 - 1

    ASSERT_TRUE(state_.IsGameOver());
    EXPECT_EQ(state_.game_result->winner, dc::Team::k1);
    EXPECT_EQ(state_.game_result->reason, dc::GameResult::Reason::kScore);
}

TEST_F(ApplyMoveTest, LeaderWinsAfterBlankLastEnd) {
    GoToLastShot(setting_.max_end - 1);
    state_.scores.team0[2] = 2;
    Throw(std::nullopt);

    ASSERT_TRUE(state_.IsGameOver());
    EXPECT_EQ(state_.game_result->winner, dc::Team::k0);
    EXPECT_EQ(state_.game_result->reason, dc::GameResult::Reason::kScore);
}

TEST_F(ApplyMoveTest, ExtraEndScoreDecidesWinner) {
    GoToLastShot(setting_.max_end);  // extra end, tied 0-0
    Throw(kTee);

    ASSERT_TRUE(state_.IsGameOver());
    EXPECT_EQ(state_.game_result->winner, dc::Team::k1);
    EXPECT_EQ(state_.game_result->reason, dc::GameResult::Reason::kScore);
}

TEST_F(ApplyMoveTest, TieAfterLastEndGoesToExtraEnd) {
    GoToLastShot(setting_.max_end - 1);
    state_.scores.team0[0] = 1;
    state_.scores.team1[1] = 1;
    Throw(std::nullopt);  // blank last end: still tied 1-1

    EXPECT_FALSE(state_.IsGameOver());
    EXPECT_EQ(state_.end, setting_.max_end);
    EXPECT_EQ(state_.shot, 0);
}

TEST_F(ApplyMoveTest, BlankTiedLastEndKeepsHammerForExtraEnd) {
    GoToLastShot(setting_.max_end - 1);
    Throw(std::nullopt);

    ASSERT_FALSE(state_.IsGameOver());
    EXPECT_EQ(state_.hammer, dc::Team::k1);
}

TEST_F(ApplyMoveTest, TeamThatTiesTheScoreLosesHammerForExtraEnd) {
    GoToLastShot(setting_.max_end - 1);
    state_.scores.team0[0] = 1;
    Throw(kTee);  // team1 scores 1: tied 1-1

    ASSERT_FALSE(state_.IsGameOver());
    EXPECT_EQ(state_.end, setting_.max_end);
    EXPECT_EQ(state_.hammer, dc::Team::k0);
}

TEST_F(ApplyMoveTest, BlankExtraEndPlaysAnotherExtraEnd) {
    GoToLastShot(setting_.max_end);  // extra end, tied 0-0
    Throw(std::nullopt);

    ASSERT_FALSE(state_.IsGameOver());
    EXPECT_EQ(state_.end, setting_.max_end + 1);
    EXPECT_EQ(state_.shot, 0);
}

TEST_F(ApplyMoveTest, SecondExtraEndScoreDecidesWinner) {
    GoToLastShot(setting_.max_end + 1);  // past the score slots, tied 0-0
    Throw(kTee);

    ASSERT_TRUE(state_.IsGameOver());
    EXPECT_EQ(state_.game_result->winner, dc::Team::k1);
    EXPECT_EQ(state_.game_result->reason, dc::GameResult::Reason::kScore);
}

TEST_F(ApplyMoveTest, TieAtExtraEndLimitIsADraw) {
    GoToLastShot(dc::GameState::kExtraEndMax - 1);
    Throw(std::nullopt);

    ASSERT_TRUE(state_.IsGameOver());
    EXPECT_EQ(state_.game_result->winner, dc::Team::kInvalid);
    EXPECT_EQ(state_.game_result->reason, dc::GameResult::Reason::kDraw);
}

}  // namespace
