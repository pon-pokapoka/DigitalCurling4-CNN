// Unit tests for include/digitalcurling/client/client_helpers.hpp

#include <gtest/gtest.h>

#include <chrono>

#include "digitalcurling/client/client_helpers.hpp"

namespace {

using digitalcurling::client::RemainingTimeFromSeconds;
using std::chrono::milliseconds;

TEST(RemainingTimeFromSecondsTest, ConvertsToMilliseconds) {
    EXPECT_EQ(RemainingTimeFromSeconds(1.5), milliseconds(1500));
    EXPECT_EQ(RemainingTimeFromSeconds(0.0), milliseconds(0));
}

TEST(RemainingTimeFromSecondsTest, NegativeTimeIsZero) {
    EXPECT_EQ(RemainingTimeFromSeconds(-0.25), milliseconds(0));
    EXPECT_EQ(RemainingTimeFromSeconds(-3600.0), milliseconds(0));
}

TEST(RemainingTimeFromSecondsTest, TimeBeyondUint32Survives) {
    EXPECT_EQ(RemainingTimeFromSeconds(5'000'000.0), milliseconds(5'000'000'000));
}

}  // namespace
