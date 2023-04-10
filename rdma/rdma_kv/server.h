#include <chrono>
#include <cstddef>
#include <thread>

#include "rome/logging/logging.h"
#include "rome/rdma/connection_manager/connection_manager.h"
#include "rome/rdma/memory_pool/memory_pool.h"
#include "kvstore.h"

class Server {
public:
  ~Server() = default;

  static void signal_handler(int signum) { 
    ROME_INFO("SIGNAL HANDLER!\n");
    exit(1);
  }

  static std::unique_ptr<Server> Create(Peer server,
                                        std::vector<Peer> clients) {
    return std::unique_ptr<Server>(new Server(server, clients));
  }

  absl::Status Launch(volatile bool *done, int runtime_s) {
    ROME_DEBUG("Starting server...");

    signal(SIGINT, signal_handler);
    
    auto status = kvstore_.Init(self_, peers_); // Starts `cm_` and connects to peers
    ROME_CHECK_OK(ROME_RETURN(status), status);

    // Sleep while clients are running if there is a set runtime.
    if (runtime_s > 0) {
      auto runtime = std::chrono::seconds();
      std::this_thread::sleep_for(runtime);
      *done = true; // Just run once
    }

    // Wait for all clients to be done.
    for (auto &p : peers_) {
      auto conn_or = pool_.connection_manager()->GetConnection(p.id);
      if (!conn_or.ok())
        return conn_or.status();

      auto *conn = conn_or.value();
      auto msg = conn->channel()->TryDeliver<AckProto>();
      while ((!msg.ok() &&
              msg.status().code() == absl::StatusCode::kUnavailable)) {
        msg = conn->channel()->TryDeliver<AckProto>();
      }
    }
    return absl::OkStatus();
  }

private:
  Server(Peer self, std::vector<Peer> peers)
      : self_(self), peers_(peers),
        pool_(self_, std::make_unique<cm_type>(self.id)), lock_(self_, pool_) {}

  const Peer self_;
  std::vector<Peer> peers_;
  X::MemoryPool pool_;
  KVStore kvstore_;
};