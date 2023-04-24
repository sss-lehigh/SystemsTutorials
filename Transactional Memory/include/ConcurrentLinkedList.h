#ifndef TRANSACTIONAL_MEMORY_CONCURRENTLINKEDLIST_H
#define TRANSACTIONAL_MEMORY_CONCURRENTLINKEDLIST_H

#include "List.h"
#include <string>

class ConcurrentSortedDoublyLinkedList : public List<Node> {
public:
  ConcurrentSortedDoublyLinkedList() : List(){};

  void insertNode(/* with */ int key) override;
  void deleteNode(/* with */ int key) override;
  [[nodiscard]] std::optional<Node *> containsNode(/* with */ int key) const override;

  void setHead(Node *newHead) {
    head = newHead;
    newHead->previous = nullptr;
  }

  static void setTail(Node *newTail) {
    newTail->next = nullptr;
  }
};

#endif // TRANSACTIONAL_MEMORY_CONCURRENTLINKEDLIST_H
