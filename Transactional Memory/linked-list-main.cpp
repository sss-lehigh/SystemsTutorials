#include <chrono>
#include <thread>
#include <utility> // std::unreachable
#include "ArgumentParser.h"
#include "LinkedList.h"
#include "Random.h"

using LinkedList = DoublyLinkedList;

void mutate(/* on */ LinkedList &list, /* slowDown */ int milliseconds) {
  switch (randomOperation()) {
  case LinkedList::Operation::Insert:
    list.insertNode(/* with */ randomValue(),
                    /* afterNodeWith */ randomValue());
    break;
  case LinkedList::Operation::Append:
    list.appendNode(/* with */ randomValue());
    break;
  case LinkedList::Operation::Delete:
    list.deleteFirstNode(/* with */ randomValue());
    break;
  }

  std::this_thread::sleep_for(std::chrono::milliseconds(milliseconds));
}

int main(int argumentCount, const char *arguments[]) {
  parse(arguments, argumentCount);

  LinkedList list{};

  while (true) {
    mutate(list, slowDown);
  }

  std::unreachable();
}
