// Copyright (c) 2023-2026 Takehiro Yoshioka
// SPDX-License-Identifier: MIT

#ifndef UCTNODE_HPP
#define UCTNODE_HPP

#include <torch/script.h>
#include <torch/cuda.h>
#include <atomic>
#include <vector>
#include <mutex>

#include "config.hpp"
#include "digitalcurling/digitalcurling.hpp"

namespace dc = digitalcurling;

using namespace config;

class UctNode
{
    public:
        typedef std::unique_ptr<UctNode> Ptr;

    public:
        UctNode();
        void    CreateChild(std::pair<int, int>);
        // void    expandChild(int);
        // void    resetAsRoot();
        // void    removeChild(Ptr);

        UctNode* GetChild(std::pair<int, int>);
        UctNode* GetParent();

        void SetGameState(dc::GameState);
        dc::GameState const& GetGameState() const;
        void SetPolicy(torch::Tensor);
        void SetFilter(std::array<std::array<std::array<bool, policy_width>, policy_weight>, policy_rotation>);
        // void SetValue(float);

        torch::Tensor GetFilter();

        // `policy` points at policy_weight * policy_width * policy_rotation floats.
        void SetEvaluatedResults(float const* policy, float value);

        void SetSimulated();
        void SetEvaluated();

        bool GetEvaluated();

        torch::Tensor GetPolicy();
        std::vector<float> const& GetPolicyValues() const;
        float GetValue();

        std::vector<std::vector<UctNode*>> const& GetChildNodeAddresses() const;

        void SetValue(float);
        void SetCount(int);
        float GetCountValue();

        int GetVisitCount();

        void SetChildCountValue(int, int, float, bool);

        // Per-action statistics. Workers read them while the updater thread writes,
        // so they are atomic; the visit count and the sum are read independently,
        // which only skews a mean by a visit in flight.
        std::vector<std::atomic<int>> const& ChildVisits() const;
        std::vector<std::atomic<float>> const& ChildSums() const;

        // Tensor copies as {1, actions}, for use once the search has stopped.
        torch::Tensor GetChildVisitCount();
        torch::Tensor GetChildSumValue();

        torch::Tensor GetChildMaxValue();
        torch::Tensor GetChildMinValue();

        std::pair<int, int> GetIndices();

        int AllocateNextChildIndex(int action);

    private:
        UctNode* parent_;
        dc::GameState       game_state_;
        std::pair<int, int> indices_;

        std::vector<float>  policy_;
        torch::Tensor       filter_;
        float   value_;

        std::atomic<int>    visit_count_;
        std::atomic<float>  sum_value_;

        std::vector<Ptr>    child_nodes_;
        std::vector<std::vector<UctNode*>> child_node_addresses_;
        std::mutex          child_reserve_mutex_;


        std::vector<std::atomic<int>>   child_visit_count_;
        std::vector<std::atomic<float>> child_sum_value_;

        // Written only by real backups, between search phases.
        std::vector<float>  child_max_value_;
        std::vector<float>  child_min_value_;

        bool    simulated_;
        bool    evaluated_;

};


#endif // UCTNODE_HPP