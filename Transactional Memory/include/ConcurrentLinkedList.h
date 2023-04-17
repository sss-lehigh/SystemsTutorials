#ifndef TRANSACTIONAL_MEMORY_CONCURRENTLINKEDLIST_H
#define TRANSACTIONAL_MEMORY_CONCURRENTLINKEDLIST_H

#include "List.h"
#include <string>

class ConcurrentDoublyLinkedList : public List {
public:
  ConcurrentDoublyLinkedList() : List(){};

  void setHead(Node *newHead) {
    head = newHead;
    newHead->previous = nullptr;
  }

  static void setTail(Node *newTail) { newTail->next = nullptr; }

  void insertNode(/* with */ int content, /* afterNodeWith */ int immediatelyPrecedingContent) const;
  void appendNode(/* with */ int content);
  void deleteFirstNode(/* with */ int content);
};

#endif // TRANSACTIONAL_MEMORY_CONCURRENTLINKEDLIST_H
