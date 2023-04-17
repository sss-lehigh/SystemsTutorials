#ifndef TRANSACTIONAL_MEMORY_LINKEDLIST_H
#define TRANSACTIONAL_MEMORY_LINKEDLIST_H

#include <string>
#include "List.h"

class SortedDoublyLinkedList : public List {
public:
  SortedDoublyLinkedList() : List(){};

  void insertNode(/* with */ int key) override;
  void deleteNode(/* with */ int key) override;
  std::optional<Node *> containsNode(/* with */ int key) const override;

  void setHead(Node *newHead) {
    head = newHead;
    if (newHead != nullptr) { newHead->previous = nullptr; }
  }

  static void setTail(Node *newTail) {
    newTail->next = nullptr;
  }
};

#endif // TRANSACTIONAL_MEMORY_LINKEDLIST_H
