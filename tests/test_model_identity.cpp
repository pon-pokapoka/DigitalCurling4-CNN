// Copyright (c) 2026 Takehiro Yoshioka
// SPDX-License-Identifier: MIT

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

#include <gtest/gtest.h>

#include "engine/model_identity.hpp"
#include "engine/sha256.hpp"

namespace {

using namespace digitalcurling::client;

std::string Sha(std::string const& s)
{
    Sha256 sha;
    sha.Update(s.data(), s.size());
    return sha.HexDigest();
}

// --- Sha256 (FIPS 180-4 test vectors) -----------------------------------------

TEST(Sha256, Empty) {
    EXPECT_EQ(Sha(""), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
}

TEST(Sha256, Abc) {
    EXPECT_EQ(Sha("abc"), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
}

TEST(Sha256, TwoBlocks) {
    EXPECT_EQ(Sha("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
              "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
}

TEST(Sha256, MillionAsInPieces) {
    Sha256 sha;
    std::string const chunk(1000, 'a');
    for (int i = 0; i < 1000; ++i) sha.Update(chunk.data(), chunk.size());
    EXPECT_EQ(sha.HexDigest(), "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
}

TEST(Sha256, File) {
    auto const path = std::filesystem::temp_directory_path() / "dc4_cnn_sha256_test.txt";
    { std::ofstream(path, std::ios::binary) << "abc"; }
    EXPECT_EQ(FileSha256(path.string()), Sha("abc"));
    std::filesystem::remove(path);
}

TEST(Sha256, MissingFileThrows) {
    EXPECT_THROW(FileSha256("/nonexistent/dc4_cnn_no_such_file"), std::runtime_error);
}

// --- Team name ----------------------------------------------------------------

ModelIdentity Identity()
{
    ModelIdentity id;
    id.model_hash = "588f425c0761aaaa";
    id.table_hash = "648e2603a02abbbb";
    return id;
}

TEST(ModelIdentity, ShortIdUsesHashPrefixes) {
    EXPECT_EQ(Identity().ShortId(), "588f425c-648e2603");
}

TEST(ComposeTeamName, ShortByDefault) {
    EXPECT_EQ(ComposeTeamName("CNNClient", false, "v2026.07-67-a604b03", Identity()), "CNNClient");
}

TEST(ComposeTeamName, FullAddsBuildAndModel) {
    EXPECT_EQ(ComposeTeamName("CNNClient", true, "v2026.07-67-a604b03", Identity()),
              "CNNClient-v2026.07-67-a604b03-588f425c-648e2603");
}

// --- EnvFlag ------------------------------------------------------------------

TEST(EnvFlag, OnlyOneOrTrue) {
    char const* name = "DC4_CNN_TEST_FLAG";
    unsetenv(name);
    EXPECT_FALSE(EnvFlag(name));
    for (char const* v : {"1", "true", "TRUE"}) {
        setenv(name, v, 1);
        EXPECT_TRUE(EnvFlag(name)) << v;
    }
    for (char const* v : {"0", "", "false", "yes"}) {
        setenv(name, v, 1);
        EXPECT_FALSE(EnvFlag(name)) << v;
    }
    unsetenv(name);
}

}  // namespace
