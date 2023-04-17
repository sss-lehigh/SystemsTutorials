#include <chrono>
#include <thread>
#include <utility> // std::unreachable
#include <vector>
#include "ArgumentParser.h"
#include "LinkedList.h"
#include "Random.h"

using LinkedList = SortedDoublyLinkedList;

void mutate(/* on */ LinkedList &list, /* slowDown */ unsigned int milliseconds) {
  /* atomic do */ {
    switch (randomOperation()) {
    case LinkedList::Operation::Insert:
      list.insertNode(/* with */ randomValue()); break;
    case LinkedList::Operation::Delete:
      list.deleteNode(/* with */ randomValue()); break;
    }
  }

  std::this_thread::sleep_for(std::chrono::milliseconds(milliseconds));
}

int main(int argumentCount, const char *arguments[]) {
  parse(arguments, argumentCount);

  LinkedList list{};

  std::vector<std::thread> threads;
  threads.reserve(threadCount);

  for (int i = 0; i < threadCount; ++i) {
    threads.emplace_back([&list] {
      while (true) {
        /* atomic do */ { mutate(list, slowDown); }
      }
    });
  }

  for (auto &thread : threads) {
    thread.join();
  }

  std::unreachable();
}
