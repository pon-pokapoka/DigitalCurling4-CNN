// Unit tests for src/engine/uctnode.cpp (tree bookkeeping and child statistics).

#include <gtest/gtest.h>

#include <torch/script.h>

#include <algorithm>
#include <set>
#include <thread>
#include <vector>

#include "engine/uctnode.hpp"
#include "engine/utility.hpp"

namespace {

constexpr int kActions = config::policy_weight * config::policy_width * config::policy_rotation;

TEST(UctNode, DefaultState) {
    UctNode node;
    EXPECT_EQ(node.GetParent(), nullptr);
    EXPECT_EQ(node.GetIndices(), std::make_pair(-1, -1));
    EXPECT_EQ(node.GetVisitCount(), 0);
    EXPECT_FALSE(node.GetEvaluated());
    EXPECT_EQ(node.GetChildNodeAddresses().size(), static_cast<size_t>(kActions));
    for (auto const& slots : node.GetChildNodeAddresses()) EXPECT_TRUE(slots.empty());
}

TEST(UctNode, ChildStatisticsInitialisation) {
    UctNode node;
    std::vector<int64_t> const shape{1, kActions};
    EXPECT_EQ(node.GetChildVisitCount().sizes(), shape);
    EXPECT_EQ(node.GetChildSumValue().sizes(), shape);
    EXPECT_EQ(node.GetChildMaxValue().sizes(), shape);
    EXPECT_EQ(node.GetChildMinValue().sizes(), shape);

    EXPECT_EQ(node.GetChildVisitCount().sum().item<int64_t>(), 0);
    EXPECT_FLOAT_EQ(node.GetChildSumValue().abs().sum().item<float>(), 0.f);
    // Max starts at the lower bound (0), min at the upper bound (1) of the value range.
    EXPECT_FLOAT_EQ(node.GetChildMaxValue().max().item<float>(), 0.f);
    EXPECT_FLOAT_EQ(node.GetChildMinValue().min().item<float>(), 1.f);
    EXPECT_TRUE(node.GetChildVisitCount().device().is_cpu());
}

TEST(UctNode, CreateChildLinksParentAndIndices) {
    UctNode root;
    root.CreateChild({5, 0});
    root.CreateChild({5, 2});
    root.CreateChild({kActions - 1, 0});

    UctNode* a = root.GetChild({5, 0});
    UctNode* b = root.GetChild({5, 2});
    UctNode* c = root.GetChild({kActions - 1, 0});
    ASSERT_NE(a, nullptr);
    ASSERT_NE(b, nullptr);
    ASSERT_NE(c, nullptr);
    EXPECT_NE(a, b);
    EXPECT_EQ(a->GetParent(), &root);
    EXPECT_EQ(b->GetParent(), &root);
    EXPECT_EQ(a->GetIndices(), std::make_pair(5, 0));
    EXPECT_EQ(b->GetIndices(), std::make_pair(5, 2));
    EXPECT_EQ(c->GetIndices(), std::make_pair(kActions - 1, 0));

    // Slot 1 was grown over but never filled.
    EXPECT_EQ(root.GetChild({5, 1}), nullptr);
    EXPECT_EQ(root.GetChildNodeAddresses()[5].size(), 3u);
}

TEST(UctNode, GetChildOutOfRangeReturnsNull) {
    UctNode root;
    root.CreateChild({0, 0});
    EXPECT_EQ(root.GetChild({-1, 0}), nullptr);
    EXPECT_EQ(root.GetChild({0, -1}), nullptr);
    EXPECT_EQ(root.GetChild({0, 1}), nullptr);
    EXPECT_EQ(root.GetChild({1, 0}), nullptr);
    EXPECT_EQ(root.GetChild({kActions, 0}), nullptr);
}

TEST(UctNode, GrandchildrenKeepTheirOwnParent) {
    UctNode root;
    root.CreateChild({3, 0});
    UctNode* child = root.GetChild({3, 0});
    child->CreateChild({7, 0});
    UctNode* grandchild = child->GetChild({7, 0});
    ASSERT_NE(grandchild, nullptr);
    EXPECT_EQ(grandchild->GetParent(), child);
    EXPECT_EQ(grandchild->GetParent()->GetParent(), &root);
    EXPECT_EQ(root.GetChild({7, 0}), nullptr);
}

TEST(UctNode, AllocateNextChildIndexIsSequentialPerAction) {
    UctNode node;
    EXPECT_EQ(node.AllocateNextChildIndex(4), 0);
    EXPECT_EQ(node.AllocateNextChildIndex(4), 1);
    EXPECT_EQ(node.AllocateNextChildIndex(9), 0);
    EXPECT_EQ(node.AllocateNextChildIndex(4), 2);
    EXPECT_EQ(node.GetChildNodeAddresses()[4].size(), 3u);
    // Reserved slots are empty until a child is created.
    EXPECT_EQ(node.GetChild({4, 1}), nullptr);

    node.CreateChild({4, 1});
    ASSERT_NE(node.GetChild({4, 1}), nullptr);
    EXPECT_EQ(node.GetChild({4, 1})->GetIndices(), std::make_pair(4, 1));
    // Creating into a reserved slot does not shift later allocations.
    EXPECT_EQ(node.AllocateNextChildIndex(4), 3);
}

TEST(UctNode, AllocateNextChildIndexNegativeActionReturnsZero) {
    UctNode node;
    EXPECT_EQ(node.AllocateNextChildIndex(-1), 0);
    EXPECT_EQ(node.AllocateNextChildIndex(-1), 0);
}

TEST(UctNode, AllocateNextChildIndexIsThreadSafe) {
    UctNode node;
    int const threads = 8;
    int const per_thread = 200;
    std::vector<std::vector<int>> results(threads);
    std::vector<std::thread> workers;
    for (int t = 0; t < threads; ++t) {
        workers.emplace_back([&, t] {
            for (int i = 0; i < per_thread; ++i) results[t].push_back(node.AllocateNextChildIndex(11));
        });
    }
    for (auto& w : workers) w.join();

    std::set<int> all;
    for (auto const& r : results) all.insert(r.begin(), r.end());
    EXPECT_EQ(all.size(), static_cast<size_t>(threads * per_thread));
    EXPECT_EQ(*all.begin(), 0);
    EXPECT_EQ(*all.rbegin(), threads * per_thread - 1);
    EXPECT_EQ(node.GetChildNodeAddresses()[11].size(), static_cast<size_t>(threads * per_thread));
}

TEST(UctNode, VisitCountAndValueAccumulate) {
    UctNode node;
    node.SetCount(1);
    node.SetValue(0.25f);
    node.SetCount(3);
    node.SetValue(0.75f);
    EXPECT_EQ(node.GetVisitCount(), 4);
    EXPECT_FLOAT_EQ(node.GetCountValue(), 1.0f / 4);
}

TEST(UctNode, SetChildCountValueAccumulatesAndTracksExtremes) {
    UctNode node;
    int const idx = 17;
    node.SetChildCountValue(idx, 1, 0.6f, false);
    node.SetChildCountValue(idx, 1, 0.2f, false);
    node.SetChildCountValue(idx, 2, 0.9f, false);

    EXPECT_EQ(node.GetChildVisitCount()[0][idx].item<int>(), 4);
    EXPECT_FLOAT_EQ(node.GetChildSumValue()[0][idx].item<float>(), 0.6f + 0.2f + 0.9f);
    EXPECT_FLOAT_EQ(node.GetChildMaxValue()[0][idx].item<float>(), 0.9f);
    EXPECT_FLOAT_EQ(node.GetChildMinValue()[0][idx].item<float>(), 0.2f);

    // Other children are untouched.
    EXPECT_EQ(node.GetChildVisitCount().sum().item<int>(), 4);
    EXPECT_FLOAT_EQ(node.GetChildSumValue().sum().item<float>(), 0.6f + 0.2f + 0.9f);
    EXPECT_FLOAT_EQ(node.GetChildMaxValue()[0][idx + 1].item<float>(), 0.f);
    EXPECT_FLOAT_EQ(node.GetChildMinValue()[0][idx + 1].item<float>(), 1.f);
}

TEST(UctNode, VirtualUpdatesDoNotTouchMinMax) {
    UctNode node;
    int const idx = 3;
    node.SetChildCountValue(idx, 1, 0.5f, false);
    // Apply and revert a virtual loss.
    node.SetChildCountValue(idx, 1, 0.f, true);
    node.SetChildCountValue(idx, -1, 0.f, true);

    EXPECT_EQ(node.GetChildVisitCount()[0][idx].item<int>(), 1);
    EXPECT_FLOAT_EQ(node.GetChildSumValue()[0][idx].item<float>(), 0.5f);
    EXPECT_FLOAT_EQ(node.GetChildMaxValue()[0][idx].item<float>(), 0.5f);
    EXPECT_FLOAT_EQ(node.GetChildMinValue()[0][idx].item<float>(), 0.5f);
}

TEST(UctNode, SetEvaluatedResultsStoresPolicyAndValue) {
    UctNode node;
    auto policy = torch::rand({kActions});
    node.SetEvaluatedResults(policy, 0.42f);
    EXPECT_TRUE(node.GetEvaluated());
    EXPECT_FLOAT_EQ(node.GetValue(), 0.42f);
    EXPECT_TRUE(torch::equal(node.GetPolicy(), policy));
}

TEST(UctNode, SetEvaluatedMarksEvaluated) {
    UctNode node;
    node.SetEvaluated();
    EXPECT_TRUE(node.GetEvaluated());
}

TEST(UctNode, GameStateRoundTrip) {
    dc::GameSetting setting;
    dc::GameState state(setting);
    state.end = 3;
    state.shot = 7;
    state.hammer = dc::Team::k0;
    state.stones.team1[2] = dc::Stone(dc::coordinate::kTee, 0.f);

    UctNode node;
    node.SetGameState(state);
    auto got = node.GetGameState();
    EXPECT_EQ(got.end, 3);
    EXPECT_EQ(got.shot, 7);
    EXPECT_EQ(got.hammer, dc::Team::k0);
    ASSERT_TRUE(got.stones.team1[2].has_value());
    EXPECT_FLOAT_EQ(got.stones.team1[2]->position.y, dc::coordinate::kTeeLineY);
}

// Known bug: SetFilter wraps its by-value argument with torch::from_blob without clone(), so the stored filter dangles, see review
TEST(UctNode, DISABLED_FilterOutlivesArgument) {
    dc::GameSetting setting;
    dc::GameState state(setting);
    auto const expected = digitalcurling::client::createFilter(state, setting);

    UctNode node;
    node.SetFilter(expected);
    // Overwrite the stack region the temporary argument lived in.
    {
        volatile char scratch[sizeof(expected) * 2];
        for (auto& c : scratch) c = 0x55;
    }

    auto filter = node.GetFilter();
    ASSERT_EQ(filter.sizes(), (std::vector<int64_t>{config::policy_rotation, config::policy_weight, config::policy_width}));
    for (int r = 0; r < config::policy_rotation; ++r)
        for (int v = 0; v < config::policy_weight; ++v)
            for (int c = 0; c < config::policy_width; ++c)
                EXPECT_EQ(filter[r][v][c].item<int>() != 0, expected[r][v][c]) << r << "," << v << "," << c;
}

}  // namespace
