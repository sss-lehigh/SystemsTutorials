cc_library(
    name = "kvstore",
    hdrs = ["kvstore.h", "server.h", "client.h", "util.h"],
    copts = ["-std=c++2a"],
    visibility = ["//visibility:public"],
    deps = [
        "@absl//absl/flags:flag",
        "@absl//absl/flags:parse",
        "@absl//absl/status",
        "@rome//rome/rdma:rdma_memory",
        "@rome//rome/rdma/channel:sync_accessor",
        "@rome//rome/rdma/connection_manager",
        "@rome//rome/rdma/memory_pool",
        "@rome//rome/rdma/memory_pool:remote_ptr",
        "@rome//rome/util:status_util",
        "@rome//rome/colosseum:client_adaptor",
        "@rome//rome/colosseum:qps_controller",
        "@rome//rome/colosseum:workload_driver",
        "@rome//rome/colosseum/streams",
    ],
)

cc_binary(
    name = "main",
    srcs = ["main.cc"],
    copts = ["-std=c++2a"],
    deps = [
        ":kvstore"
    ],
)