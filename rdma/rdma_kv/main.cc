#include <chrono>
#include <cstddef>
#include <thread>

#include "rome/logging/logging.h"
#include "rome/rdma/memory_pool/memory_pool.h"


//! Should i not be using Rome? I think its a super valuable tool to use that breaks down the barriers to using rdma, but not sure how widely distributed it's allowed to be? 