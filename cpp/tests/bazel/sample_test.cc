#include <gtest/gtest.h>

/*
This test suite is part of the bazel tutorial. It works with the following CLI:
bazel test --cxxopt=-std=c++14 --test_output=all //:sample_test

See:
https://google.github.io/googletest/quickstart-bazel.html

*/

// Demonstrate some basic assertions.
TEST(HelloTest, BasicAssertions) {
  // Expect two strings not to be equal.
  EXPECT_STRNE("hello", "world");
  // Expect equality.
  EXPECT_EQ(7 * 6, 42);
}