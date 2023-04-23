#include <chrono>
#include <thread>
#include <utility> // std::unreachable
#include "ArgumentParser.h"
#include "LinkedList.h"
#include "Random.h"

using LinkedList = SortedDoublyLinkedList;

void mutate(LinkedList &list) {
  auto operation = randomOperation();
  auto key = randomValue();

  switch (operation) {
  case LinkedList::Operation::Insert:
    list.insertNode(/* with */ key); break;
  case LinkedList::Operation::Delete:
    list.deleteNode(/* with */ key); break;
  }

  std::this_thread::sleep_for(std::chrono::milliseconds(slowDownMilliseconds));
}

int main(int argumentCount, const char *arguments[]) {
  parse(arguments, argumentCount);

  LinkedList list{};

  while (true) {
    mutate(list);
  }

  std::unreachable();
}
