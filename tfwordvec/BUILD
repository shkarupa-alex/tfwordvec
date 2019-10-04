package(default_visibility = ["//visibility:public"])

cc_binary(
    name = 'python/ops/_tfwordvec_ops.so',
    srcs = [
        "cc/kernels/down_sample.cc",
        "cc/ops/down_sample.cc",

        "cc/kernels/skip_gram.cc",
        "cc/ops/skip_gram.cc",

        "cc/kernels/c_b_o_w.cc",
        "cc/ops/c_b_o_w.cc",

#        "cc/kernels/negative_sampling.cc",
#        "cc/ops/negative_sampling.cc",
    ],
    linkshared = 1,
    deps = [
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
    copts = ["-pthread", "-std=c++11"]
)

py_library(
    name = "tfwordvec_py",
    srcs = ([
        "__init__.py",
        "estimator.py",
        "input.py",
        "model.py",
        "train.py",
        "python/__init__.py",
        "python/ops/__init__.py",
    ]),
    data = [":python/ops/_tfwordvec_ops.so"],
    srcs_version = "PY2AND3",
)

py_test(
    name = "down_sample_test",
    srcs = ["python/kernel_tests/down_sample_test.py"],
    deps = [":tfwordvec_py"],
    srcs_version = "PY2AND3",
)

py_test(
    name = "skip_gram_test",
    srcs = ["python/kernel_tests/skip_gram_test.py"],
    deps = [":tfwordvec_py"],
    srcs_version = "PY2AND3",
)

py_test(
    name = "c_b_o_w_test",
    srcs = ["python/kernel_tests/c_b_o_w_test.py"],
    deps = [":tfwordvec_py"],
    srcs_version = "PY2AND3",
)

#py_test(
#    name = "negative_sampling_test",
#    srcs = ["python/kernel_tests/negative_sampling_test.py"],
#    deps = [":tfwordvec_py"],
#    srcs_version = "PY2AND3",
#)
