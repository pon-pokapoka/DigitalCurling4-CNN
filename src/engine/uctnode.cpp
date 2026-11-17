// Copyright (c) 2023-2026 Takehiro Yoshioka
// SPDX-License-Identifier: MIT

#include "uctnode.hpp"

#include <algorithm>

UctNode::UctNode():
parent_(nullptr),
game_state_(),
indices_(std::make_pair(-1, -1)),
evaluated_(false),
simulated_(false),
value_(-1),
visit_count_(0),
sum_value_(0),
child_nodes_(),
child_node_addresses_(policy_weight*policy_width*policy_rotation),
child_visit_count_(policy_weight*policy_width*policy_rotation),
child_sum_value_(policy_weight*policy_width*policy_rotation),
child_max_value_(policy_weight*policy_width*policy_rotation, 0.f),
child_min_value_(policy_weight*policy_width*policy_rotation, 1.f)
{}

void UctNode::CreateChild(std::pair<int, int> indices) {
    std::unique_ptr<UctNode> child(new UctNode());
    child->parent_ = this;
    child->indices_ = indices;

    auto& action_children = child_node_addresses_[indices.first];
    if (indices.second >= static_cast<int>(action_children.size())) {
        action_children.resize(indices.second + 1, nullptr);
    }
    action_children[indices.second] = child.get();

    child_nodes_.push_back(std::move(child));
}

// void UctNode::expandChild(int childData) {
//     std::unique_ptr<UctNode> newChild = new UctNode(childData);
//     addChild(newChild);
// }

// void UctNode::resetAsRoot() {
    // if (parent_ != nullptr) {
    //     parent_->removeChild(this);
    //     parent_ = nullptr;
    // }
// }

// void UctNode::removeChild(std::unique_ptr<UctNode> child) {
//     auto it = std::find(child_nodes_.begin(), child_nodes_.end(), child);
//     if (it != child_nodes_.end()) {
//         child_nodes_.erase(it);
//     }
// }

UctNode* UctNode::GetChild(std::pair<int, int> indices)
{
    if (indices.first < 0 || indices.first >= static_cast<int>(child_node_addresses_.size())) {
        return nullptr;
    }

    auto const& action_children = child_node_addresses_[indices.first];
    if (indices.second < 0 || indices.second >= static_cast<int>(action_children.size())) {
        return nullptr;
    }
    return action_children[indices.second];
}


UctNode* UctNode::GetParent()
{
    return parent_;
}

int UctNode::AllocateNextChildIndex(int action) {
    std::lock_guard<std::mutex> lock(child_reserve_mutex_);
    if (action < 0) return 0;
    if (action >= static_cast<int>(child_node_addresses_.size())) {
        child_node_addresses_.resize(action + 1);
    }
    int idx = static_cast<int>(child_node_addresses_[action].size());
    child_node_addresses_[action].resize(idx + 1, nullptr);
    return idx;
}

void UctNode::SetGameState(dc::GameState game_state)
{
    game_state_ = game_state;
    simulated_ = true;
}

dc::GameState const& UctNode::GetGameState() const
{
    return game_state_;
}


void UctNode::SetPolicy(torch::Tensor policy)
{
    auto const flat = policy.to(torch::kCPU).to(torch::kFloat).contiguous().reshape({-1});
    float const* data = flat.data_ptr<float>();
    policy_.assign(data, data + flat.numel());
}


void UctNode::SetFilter(std::array<std::array<std::array<bool, policy_width>, policy_weight>, policy_rotation> filter)
{
    // from_blob only wraps the argument's storage, which dies on return: clone it.
    filter_ = torch::from_blob(filter.data(), {policy_rotation, policy_weight, policy_width}, torch::kBool).clone();
}

// void UctNode::SetValue(float value)
// {
//     value_ = value;
// }

torch::Tensor UctNode::GetFilter()
{
    return filter_;
}

void UctNode::SetEvaluatedResults(float const* policy, float value)
{
    policy_.assign(policy, policy + policy_weight*policy_width*policy_rotation);
    value_ = value;
    evaluated_ = true;
}

void UctNode::SetSimulated()
{
    simulated_ = true;
}

void UctNode::SetEvaluated()
{
    evaluated_ = true;
}

bool UctNode::GetEvaluated()
{
    return evaluated_;
}

torch::Tensor UctNode::GetPolicy()
{
    return torch::tensor(policy_, torch::kFloat);
}

std::vector<float> const& UctNode::GetPolicyValues() const
{
    return policy_;
}

float UctNode::GetValue()
{
    return value_;
}

std::vector<std::vector<UctNode*>> const& UctNode::GetChildNodeAddresses() const
{
    return child_node_addresses_;
}

void UctNode::SetValue(float value)
{
    sum_value_.fetch_add(value, std::memory_order_relaxed);
}

void UctNode::SetCount(int count)
{
    visit_count_.fetch_add(count, std::memory_order_relaxed);
}

float UctNode::GetCountValue()
{
    return sum_value_.load(std::memory_order_relaxed) / visit_count_.load(std::memory_order_relaxed);
}

int UctNode::GetVisitCount()
{
    return visit_count_.load(std::memory_order_relaxed);
}

void UctNode::SetChildCountValue(int index, int count, float value, bool update_virtual)
{
    child_visit_count_[index].fetch_add(count, std::memory_order_relaxed);
    child_sum_value_[index].fetch_add(value, std::memory_order_relaxed);

    if (!update_virtual) {
        child_max_value_[index] = std::max(child_max_value_[index], value);
        child_min_value_[index] = std::min(child_min_value_[index], value);
    }
}

std::vector<std::atomic<int>> const& UctNode::ChildVisits() const
{
    return child_visit_count_;
}

std::vector<std::atomic<float>> const& UctNode::ChildSums() const
{
    return child_sum_value_;
}

namespace {

// Copy per-action values into a {1, actions} tensor. Atomics convert on read.
template <class Values, class Element>
torch::Tensor ToTensor(Values const& values, torch::ScalarType type)
{
    std::vector<Element> copy(values.begin(), values.end());
    return torch::tensor(copy, type).reshape({1, static_cast<int64_t>(copy.size())});
}

}  // namespace

torch::Tensor UctNode::GetChildVisitCount()
{
    return ToTensor<decltype(child_visit_count_), int>(child_visit_count_, torch::kInt);
}

torch::Tensor UctNode::GetChildSumValue()
{
    return ToTensor<decltype(child_sum_value_), float>(child_sum_value_, torch::kFloat);
}

torch::Tensor UctNode::GetChildMaxValue()
{
    return ToTensor<decltype(child_max_value_), float>(child_max_value_, torch::kFloat);
}

torch::Tensor UctNode::GetChildMinValue()
{
    return ToTensor<decltype(child_min_value_), float>(child_min_value_, torch::kFloat);
}

std::pair<int, int> UctNode::GetIndices()
{
    return indices_;
}