// Copyright (c) 2023-2026 Takehiro Yoshioka
// SPDX-License-Identifier: MIT

// Move-selection rules used by the tree search, kept free of search state so
// they can be unit-tested without a model.
//
// Every value stored in the tree is the win probability of the searching team
// (Skip::team_), in [0, 1]. "my_turn" means the searching team chooses at the
// node; otherwise the opponent does and values are seen as 1 - v.

#ifndef SEARCH_POLICY_HPP
#define SEARCH_POLICY_HPP

#include <algorithm>
#include <climits>
#include <cmath>
#include <limits>
#include <random>

#include <torch/torch.h>

#include "config.hpp"
#include "utility.hpp"

namespace digitalcurling::client::search_policy {

constexpr int kNumActions = config::policy_rotation * config::policy_weight * config::policy_width;

// Legal shots over the policy grid as a {1, kNumActions} bool tensor.
inline torch::Tensor LegalMask()
{
    auto const filt = createFilter();
    auto mask = torch::zeros({1, kNumActions}, torch::kBool);
    auto acc = mask.accessor<bool, 2>();
    for (int a = 0; a < kNumActions; ++a) acc[0][a] = filt[a];
    return mask;
}

// PUCT score from the perspective of the side choosing at the node.
// An unvisited child counts as a loss (q = 0) for whoever is choosing, so both
// teams get the same pessimistic first-play value; the exploration term u is
// largest for unvisited children, so high-prior shots are still tried first.
// Illegal shots score -inf.
inline torch::Tensor PuctScore(torch::Tensor const& child_visits,
                               torch::Tensor const& child_sums,
                               torch::Tensor const& policy,
                               torch::Tensor const& legal,
                               bool my_turn,
                               float c_puct)
{
    auto n = child_visits.to(torch::kFloat);
    float const total = n.sum().item<float>();
    auto q = child_sums.to(torch::kFloat) / n;   // NaN where n == 0; replaced below
    if (!my_turn) q = 1 - q;
    q = torch::where(n > 0, q, torch::zeros_like(q));
    auto u = policy.to(torch::kFloat) * std::sqrt(total + 1) / (1 + n);
    auto score = q + c_puct * u;
    return torch::where(legal, score, torch::full_like(score, -std::numeric_limits<float>::infinity()));
}

// Value to add to a child's value sum for `count` virtual-loss visits
// (count < 0 undoes them). Virtual loss must look like a loss to the side
// choosing at the node, so other threads avoid the path: 0 (loss for us) at
// our nodes, 1 per visit (loss for the opponent) at opponent nodes.
inline float VirtualLossValue(bool my_turn, int count)
{
    return my_turn ? 0.f : static_cast<float>(count);
}

// Root exploration: the legal action with the fewest visits, ties broken
// uniformly at random, so every legal shot is tried once before any is tried
// twice. Returns -1 if no action is legal.
inline int LeastVisitedLegalAction(torch::Tensor const& child_visits,
                                   torch::Tensor const& legal,
                                   std::mt19937& gen)
{
    auto visits = child_visits.to(torch::kInt).contiguous();
    auto mask = legal.to(torch::kBool).contiguous();
    int const* v = visits.data_ptr<int>();
    bool const* l = mask.data_ptr<bool>();
    int const n = static_cast<int>(visits.numel());

    int best = -1;
    int best_visits = INT_MAX;
    int ties = 0;
    for (int a = 0; a < n; ++a) {
        if (!l[a]) continue;
        if (v[a] < best_visits) {
            best_visits = v[a];
            best = a;
            ties = 1;
        } else if (v[a] == best_visits) {
            // reservoir sampling: each tied action ends up chosen with probability 1/ties
            ++ties;
            if (std::uniform_int_distribution<int>(0, ties - 1)(gen) == 0) best = a;
        }
    }
    return best;
}

// Minimum visits a root move needs to be chosen as the final move: at least
// config::min_visit and config::min_visit_ratio of the most-visited move, but
// never more than the most-visited move has, so that move always qualifies.
// Returns at least 1 so unvisited moves (value sum / 0) never qualify.
inline int FinalVisitThreshold(int max_child_visits)
{
    int const relative = static_cast<int>(std::ceil(config::min_visit_ratio * max_child_visits));
    int const threshold = std::max(config::min_visit, relative);
    return std::max(1, std::min(threshold, max_child_visits));
}

} // namespace digitalcurling::client::search_policy

#endif // SEARCH_POLICY_HPP
