// Unit tests for src/engine/readcsv.hpp

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <string>

#include "engine/readcsv.hpp"

namespace {

namespace fs = std::filesystem;

class ReadCsvTest : public ::testing::Test {
protected:
    void SetUp() override {
        auto const* info = ::testing::UnitTest::GetInstance()->current_test_info();
        dir_ = fs::path(::testing::TempDir()) / (std::string("readcsv_") + info->name());
        fs::create_directories(dir_);
    }

    void TearDown() override {
        std::error_code ec;
        fs::remove_all(dir_, ec);
    }

    std::string WriteFile(std::string const& name, std::string const& content) {
        fs::path path = dir_ / name;
        std::ofstream out(path, std::ios::binary);
        out << content;
        return path.string();
    }

    fs::path dir_;
};

TEST_F(ReadCsvTest, ParsesRowsAndColumns) {
    auto path = WriteFile("a.csv", "1,2,3\n4,5,6\n");
    auto data = readcsv(path);

    ASSERT_EQ(data.size(), 2u);
    EXPECT_EQ(data[0], (std::vector<double>{1, 2, 3}));
    EXPECT_EQ(data[1], (std::vector<double>{4, 5, 6}));
}

TEST_F(ReadCsvTest, LastLineWithoutTrailingNewline) {
    auto path = WriteFile("a.csv", "1,2\n3,4");
    auto data = readcsv(path);

    ASSERT_EQ(data.size(), 2u);
    EXPECT_EQ(data[1], (std::vector<double>{3, 4}));
}

TEST_F(ReadCsvTest, ParsesNegativeFractionalAndScientificValues) {
    auto path = WriteFile("a.csv", "-0.5,1e-3,2.5E2, 7\n");
    auto data = readcsv(path);

    ASSERT_EQ(data.size(), 1u);
    ASSERT_EQ(data[0].size(), 4u);
    EXPECT_DOUBLE_EQ(data[0][0], -0.5);
    EXPECT_DOUBLE_EQ(data[0][1], 1e-3);
    EXPECT_DOUBLE_EQ(data[0][2], 250.0);
    EXPECT_DOUBLE_EQ(data[0][3], 7.0);  // leading whitespace is accepted
}

TEST_F(ReadCsvTest, CustomDelimiter) {
    auto path = WriteFile("a.tsv", "1\t2\n3\t4\n");
    auto data = readcsv(path, '\t');

    ASSERT_EQ(data.size(), 2u);
    EXPECT_EQ(data[0], (std::vector<double>{1, 2}));
    EXPECT_EQ(data[1], (std::vector<double>{3, 4}));
}

TEST_F(ReadCsvTest, RaggedRowsKeepTheirLengths) {
    auto path = WriteFile("a.csv", "1\n2,3,4\n");
    auto data = readcsv(path);

    ASSERT_EQ(data.size(), 2u);
    EXPECT_EQ(data[0].size(), 1u);
    EXPECT_EQ(data[1].size(), 3u);
}

TEST_F(ReadCsvTest, EmptyFileGivesEmptyTable) {
    auto path = WriteFile("empty.csv", "");
    EXPECT_TRUE(readcsv(path).empty());
}

TEST_F(ReadCsvTest, NonNumericTokenThrows) {
    auto path = WriteFile("bad.csv", "1,abc\n");
    EXPECT_ANY_THROW(readcsv(path));
}

TEST_F(ReadCsvTest, BlankLinesAreSkipped) {
    auto path = WriteFile("a.csv", "1,2\n\n3,4\n\n");
    auto data = readcsv(path);

    ASSERT_EQ(data.size(), 2u);
    EXPECT_EQ(data[0], (std::vector<double>{1, 2}));
    EXPECT_EQ(data[1], (std::vector<double>{3, 4}));
}

TEST_F(ReadCsvTest, MissingFileIsAnError) {
    fs::path path = dir_ / "does_not_exist.csv";
    ASSERT_FALSE(fs::exists(path));
    EXPECT_THROW(readcsv(path.string()), std::runtime_error);
}

TEST(RequireTableShapeTest, AcceptsLargeEnoughTable) {
    std::vector<std::vector<double>> const table{{1, 2, 3}, {4, 5, 6}};
    EXPECT_NO_THROW(requireTableShape(table, 2, 3, "t"));
    EXPECT_NO_THROW(requireTableShape(table, 1, 2, "t"));
}

TEST(RequireTableShapeTest, RejectsMissingRows) {
    std::vector<std::vector<double>> const table{{1, 2}};
    EXPECT_THROW(requireTableShape(table, 2, 2, "t"), std::runtime_error);
}

TEST(RequireTableShapeTest, RejectsShortRow) {
    std::vector<std::vector<double>> const table{{1, 2}, {3}};
    EXPECT_THROW(requireTableShape(table, 2, 2, "t"), std::runtime_error);
}

TEST(RequireTableShapeTest, RejectsEmptyTable) {
    EXPECT_THROW(requireTableShape({}, 1, 1, "t"), std::runtime_error);
}

}  // namespace
