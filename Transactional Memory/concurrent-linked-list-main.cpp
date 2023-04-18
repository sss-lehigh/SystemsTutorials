#include <chrono>
#include <thread>
#include <utility> // std::unreachable
#include <vector>
#include "ArgumentParser.h"
#include "ConcurrentLinkedList.h"
#include "Random.h"

using LinkedList = ConcurrentSortedDoublyLinkedList;

void mutate(/* on */ LinkedList &list, /* slowDown */ unsigned int milliseconds) {
  auto operation = randomOperation();
  auto key = randomValue();

  switch (operation) {
  case LinkedList::Operation::Insert:
    list.insertNode(/* with */ key); break;
  case LinkedList::Operation::Delete:
    list.deleteNode(/* with */ key); break;
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
        mutate(list, slowDown);
      }
    });
  }

  for (auto &thread : threads) {
    thread.join();
  }

  std::unreachable();
}
