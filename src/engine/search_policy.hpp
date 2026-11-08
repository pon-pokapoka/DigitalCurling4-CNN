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
#include <vector>

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

// Indices of the true entries of a {1, kNumActions} bool mask.
inline std::vector<int> LegalActions(torch::Tensor const& legal)
{
    auto mask = legal.to(torch::kBool).contiguous();
    bool const* data = mask.data_ptr<bool>();
    std::vector<int> actions;
    for (int a = 0; a < static_cast<int>(mask.numel()); ++a) {
        if (data[a]) actions.push_back(a);
    }
    return actions;
}

// PUCT score from the perspective of the side choosing at the node. This tensor
// form is the reference that BestPuctAction, used by the search, is tested against.
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

// The legal action with the highest PuctScore, without tensors: this runs at
// every tree level of every descent. `child_visits` and `child_sums` are indexed
// by action and may be atomics. Returns -1 if no action is legal.
template <class Visits, class Sums>
inline int BestPuctAction(Visits const& child_visits,
                          Sums const& child_sums,
                          float const* policy,
                          std::vector<int> const& legal_actions,
                          bool my_turn,
                          float c_puct)
{
    int total = 0;
    for (int a : legal_actions) total += child_visits[a];
    float const sqrt_total = std::sqrt(static_cast<float>(total) + 1);

    int best = -1;
    float best_score = -std::numeric_limits<float>::infinity();
    for (int a : legal_actions) {
        float const n = static_cast<float>(int(child_visits[a]));
        float q = 0.f;
        if (n > 0) {
            q = float(child_sums[a]) / n;
            if (!my_turn) q = 1 - q;
        }
        float const score = q + c_puct * (policy[a] * sqrt_total / (1 + n));
        if (score > best_score) {
            best_score = score;
            best = a;
        }
    }
    return best;
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
template <class Visits>
inline int LeastVisitedLegalAction(Visits const& child_visits,
                                   std::vector<int> const& legal_actions,
                                   std::mt19937& gen)
{
    int best = -1;
    int best_visits = INT_MAX;
    int ties = 0;
    for (int a : legal_actions) {
        int const visits = child_visits[a];
        if (visits < best_visits) {
            best_visits = visits;
            best = a;
            ties = 1;
        } else if (visits == best_visits) {
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

// The final move: the highest mean value among legal moves with at least
// FinalVisitThreshold of the most-visited move's visits; ties go to the move
// with more visits. Returns -1 if no legal move was visited.
template <class Visits, class Sums>
inline int FinalMove(Visits const& child_visits, Sums const& child_sums, std::vector<int> const& legal_actions)
{
    int max_visits = 0;
    for (int a : legal_actions) max_visits = std::max(max_visits, static_cast<int>(child_visits[a]));
    if (max_visits == 0) return -1;
    int const threshold = FinalVisitThreshold(max_visits);

    int best = -1;
    float best_mean = 0.f;
    int best_visits = 0;
    for (int a : legal_actions) {
        int const visits = child_visits[a];
        if (visits < threshold) continue;
        float const mean = float(child_sums[a]) / visits;
        if (best < 0 || mean > best_mean || (mean == best_mean && visits > best_visits)) {
            best = a;
            best_mean = mean;
            best_visits = visits;
        }
    }
    return best;
}

} // namespace digitalcurling::client::search_policy

#endif // SEARCH_POLICY_HPP
