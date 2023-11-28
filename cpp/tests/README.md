To get the tests working I tried several approaches:

**Bazel Approach:**
From [their page](https://bazel.build/about/intro): Bazel is an open-source build and test tool similar to Make, Maven, and Gradle. It uses a human-readable, high-level build language. Bazel supports projects in multiple languages and builds outputs for multiple platforms. Bazel supports large codebases across multiple repositories, and large numbers of users.

I installed the library following the instructions [Here](https://bazel.build/install/ubuntu)

It worked great for `tests/bazel/sample_test.cc`

However, it won't work with CUDA code, so I tried the solution in `rules_cuda` [repository](https://github.com/bazel-contrib/rules_cuda) which is located in the `basic` directory. I didn't manage to get it working, though.

**chatGPT Approach:**
I asked chatGPT how to write a CUDA test and it gave me what is now in `tests/chatGPT/test_file.cu` that worked out of the box.
