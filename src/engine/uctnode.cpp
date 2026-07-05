#include "uctnode.hpp"

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
child_visit_count_(torch::zeros({1, policy_weight*policy_width*policy_rotation}, torch::kInt)),
child_sum_value_(torch::zeros({1, policy_weight*policy_width*policy_rotation}, torch::kFloat)),
child_max_value_(torch::zeros({1, policy_weight*policy_width*policy_rotation}, torch::kFloat)),
child_min_value_(torch::ones({1, policy_weight*policy_width*policy_rotation}, torch::kFloat))
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

dc::GameState UctNode::GetGameState()
{
    return game_state_;
}


void UctNode::SetPolicy(torch::Tensor policy)
{
    policy_ = policy;
}


void UctNode::SetFilter(std::array<std::array<std::array<bool, policy_width>, policy_weight>, policy_rotation> filter)
{
    filter_ = torch::from_blob(filter.data(), {policy_rotation, policy_weight, policy_width}, torch::kInt8);
}

// void UctNode::SetValue(float value)
// {
//     value_ = value;
// }

torch::Tensor UctNode::GetFilter()
{
    return filter_;
}

void UctNode::SetEvaluatedResults(torch::Tensor policy, float value)
{
    policy_ = policy;
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
    sum_value_ += value;
}

void UctNode::SetCount(int count)
{
    visit_count_ += count;
}

float UctNode::GetCountValue()
{
    return sum_value_ / visit_count_;
}

int UctNode::GetVisitCount()
{
    return visit_count_;
}

void UctNode::SetChildCountValue(int index, int count, float value, bool update_virtual)
{
    child_visit_count_.index({0, index}) += count;
    child_sum_value_.index({0, index}) += value;

    if (!update_virtual) {
        if (child_max_value_.index({0, index}).item<float>() < value) {
            child_max_value_.index_put_({0, index}, value);
        }
        if (child_min_value_.index({0, index}).item<float>() > value) {
            child_min_value_.index_put_({0, index}, value);
        }
    }
}

torch::Tensor UctNode::GetChildVisitCount()
{
    return child_visit_count_;
}

torch::Tensor UctNode::GetChildSumValue()
{
    return child_sum_value_;
}

torch::Tensor UctNode::GetChildMaxValue()
{
    return child_max_value_;
}

torch::Tensor UctNode::GetChildMinValue()
{
    return child_min_value_;
}

std::pair<int, int> UctNode::GetIndices()
{
    return indices_;
}