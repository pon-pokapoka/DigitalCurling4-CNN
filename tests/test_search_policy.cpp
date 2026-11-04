// Unit tests for src/engine/search_policy.hpp (move-selection rules of the tree search).

#include <gtest/gtest.h>

#include <torch/script.h>

#include <cmath>
#include <limits>
#include <random>
#include <set>

#include "engine/search_policy.hpp"

namespace {

namespace sp = digitalcurling::client::search_policy;
constexpr int kActions = sp::kNumActions;

torch::Tensor Zeros() { return torch::zeros({1, kActions}, torch::kFloat); }
torch::Tensor Visits() { return torch::zeros({1, kActions}, torch::kInt); }
torch::Tensor AllLegal() { return torch::ones({1, kActions}, torch::kBool); }

float At(torch::Tensor const& t, int a) { return t.index({0, a}).item<float>(); }

// --- LegalMask ---------------------------------------------------------------

TEST(LegalMask, MatchesCreateFilter) {
    auto const filt = digitalcurling::client::createFilter();
    auto const mask = sp::LegalMask();
    ASSERT_EQ(mask.sizes(), (std::vector<int64_t>{1, kActions}));
    ASSERT_EQ(mask.scalar_type(), torch::kBool);
    int legal = 0;
    for (int a = 0; a < kActions; ++a) {
        EXPECT_EQ(mask.index({0, a}).item<bool>(), filt[a]) << a;
        legal += filt[a];
    }
    EXPECT_GT(legal, 0);
    EXPECT_LT(legal, kActions);
}

// --- PuctScore ---------------------------------------------------------------

TEST(PuctScore, UnvisitedChildIsALossForWhoeverChooses) {
    // With no prior, the score is just q: an untried shot must be 0 for both sides.
    auto const policy = Zeros();
    for (bool my_turn : {true, false}) {
        auto const score = sp::PuctScore(Visits(), Zeros(), policy, AllLegal(), my_turn, config::c_puct);
        EXPECT_FLOAT_EQ(At(score, 0), 0.f) << "my_turn=" << my_turn;
        EXPECT_FLOAT_EQ(At(score, kActions - 1), 0.f) << "my_turn=" << my_turn;
    }
}

TEST(PuctScore, VisitedChildIsSeenFromTheChoosersSide) {
    auto visits = Visits();
    auto sums = Zeros();
    visits.index_put_({0, 5}, 10);
    sums.index_put_({0, 5}, 7.f);  // our win probability 0.7
    auto const policy = Zeros();
    EXPECT_NEAR(At(sp::PuctScore(visits, sums, policy, AllLegal(), true, config::c_puct), 5), 0.7f, 1e-6);
    EXPECT_NEAR(At(sp::PuctScore(visits, sums, policy, AllLegal(), false, config::c_puct), 5), 0.3f, 1e-6);
}

TEST(PuctScore, OpponentPrefersAGoodExploredReplyOverAnUntriedOne) {
    // Regression: unvisited children used to score 1 - 0 = 1 at opponent nodes,
    // beating every explored reply. Equal priors, so only q decides.
    auto visits = Visits();
    auto sums = Zeros();
    visits.index_put_({0, 1}, 4);
    sums.index_put_({0, 1}, 1.6f);  // our win prob 0.4, i.e. 0.6 for the opponent
    auto policy = Zeros();
    policy.index_put_({0, 1}, 1e-4f);
    policy.index_put_({0, 2}, 1e-4f);  // untried, same prior
    auto const score = sp::PuctScore(visits, sums, policy, AllLegal(), false, config::c_puct);
    EXPECT_GT(At(score, 1), At(score, 2));
}

TEST(PuctScore, ExplorationTermForUntriedChild) {
    auto visits = Visits();
    visits.index_put_({0, 0}, 99);  // N = 99, so sqrt(N + 1) = 10
    auto policy = Zeros();
    policy.index_put_({0, 3}, 0.25f);
    auto const score = sp::PuctScore(visits, Zeros(), policy, AllLegal(), true, config::c_puct);
    EXPECT_NEAR(At(score, 3), config::c_puct * 0.25f * 10.f, 1e-4);
}

TEST(PuctScore, IllegalShotsNeverWin) {
    auto legal = AllLegal();
    legal.index_put_({0, 7}, false);
    auto policy = Zeros();
    policy.index_put_({0, 7}, 1.f);  // the CNN loves an illegal shot
    auto const score = sp::PuctScore(Visits(), Zeros(), policy, legal, true, config::c_puct);
    EXPECT_EQ(At(score, 7), -std::numeric_limits<float>::infinity());
    EXPECT_NE(std::get<1>(torch::topk(score, 1)).index({0, 0}).item<int>(), 7);
}

TEST(PuctScore, IgnoresPolicyDtype) {
    auto policy = Zeros();
    policy.index_put_({0, 3}, 0.5f);
    auto const f = sp::PuctScore(Visits(), Zeros(), policy, AllLegal(), true, config::c_puct);
    auto const b = sp::PuctScore(Visits(), Zeros(), policy.to(torch::kBFloat16), AllLegal(), true, config::c_puct);
    EXPECT_EQ(f.scalar_type(), torch::kFloat);
    EXPECT_NEAR(At(f, 3), At(b, 3), 1e-2);
}

// --- VirtualLossValue --------------------------------------------------------

TEST(VirtualLoss, MakesThePathLessAttractiveForTheChooser) {
    for (bool my_turn : {true, false}) {
        auto visits = Visits();
        auto sums = Zeros();
        visits.index_put_({0, 4}, 10);
        sums.index_put_({0, 4}, 5.f);  // q = 0.5 for both sides
        auto const policy = Zeros();
        float const before = At(sp::PuctScore(visits, sums, policy, AllLegal(), my_turn, config::c_puct), 4);

        int const vl = config::virtual_loss;
        visits.index_put_({0, 4}, 10 + vl);
        sums.index_put_({0, 4}, 5.f + sp::VirtualLossValue(my_turn, vl));
        float const during = At(sp::PuctScore(visits, sums, policy, AllLegal(), my_turn, config::c_puct), 4);
        EXPECT_LT(during, before) << "my_turn=" << my_turn;

        visits.index_put_({0, 4}, 10);
        sums.index_put_({0, 4}, 5.f + sp::VirtualLossValue(my_turn, vl) + sp::VirtualLossValue(my_turn, -vl));
        EXPECT_FLOAT_EQ(At(sp::PuctScore(visits, sums, policy, AllLegal(), my_turn, config::c_puct), 4), before)
            << "undo must restore the score, my_turn=" << my_turn;
    }
}

// --- LeastVisitedLegalAction -------------------------------------------------

TEST(LeastVisitedLegalAction, PicksTheLeastVisitedLegalShot) {
    std::mt19937 gen(1);
    auto visits = torch::full({1, kActions}, 5, torch::kInt);
    auto legal = AllLegal();
    visits.index_put_({0, 10}, 1);
    visits.index_put_({0, 11}, 0);
    legal.index_put_({0, 11}, false);  // fewer visits, but illegal
    EXPECT_EQ(sp::LeastVisitedLegalAction(visits, legal, gen), 10);
}

TEST(LeastVisitedLegalAction, NoLegalShot) {
    std::mt19937 gen(1);
    EXPECT_EQ(sp::LeastVisitedLegalAction(Visits(), torch::zeros({1, kActions}, torch::kBool), gen), -1);
}

TEST(LeastVisitedLegalAction, TriesEveryLegalShotOnceBeforeAnyTwice) {
    std::mt19937 gen(42);
    auto const legal = sp::LegalMask();
    int const n_legal = legal.sum().item<int>();
    auto visits = Visits();
    std::set<int> seen;
    for (int k = 0; k < n_legal; ++k) {
        int const a = sp::LeastVisitedLegalAction(visits, legal, gen);
        ASSERT_GE(a, 0);
        ASSERT_TRUE(legal.index({0, a}).item<bool>()) << a;
        ASSERT_TRUE(seen.insert(a).second) << "shot " << a << " tried twice after " << k << " picks";
        visits.index_put_({0, a}, visits.index({0, a}).item<int>() + 1);
    }
    EXPECT_EQ(static_cast<int>(seen.size()), n_legal);
}

TEST(LeastVisitedLegalAction, BreaksTiesRandomly) {
    std::mt19937 gen(7);
    std::set<int> picks;
    for (int k = 0; k < 200; ++k) picks.insert(sp::LeastVisitedLegalAction(Visits(), AllLegal(), gen));
    EXPECT_GT(picks.size(), 100u);  // not always the first index
}

}  // namespace
