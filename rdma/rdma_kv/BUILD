package(default_visibility = ["//visibility:public"])

cc_library(
    name = "util",
    hdrs = ["util.h"],
    visibility = ["//visibility:public"],
    deps = [],
)

cc_library(
    name = "kvstore",
    hdrs = ["kvstore.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@absl//absl/strings",
    ],
)

cc_library(
    name = "server",
    hdrs = ["server.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":kvstore",
        "@absl//absl/strings",
    ],
)

cc_library(
    name = "client",
    hdrs = ["client.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":kvstore",
        "@absl//absl/strings",
    ],
)

cc_library(
    name = "main",
    hdrs = ["main.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":kvstore",
        ":server",
        ":client",
        "@absl//absl/strings",
    ],
)






