#ifndef TRANSACTIONAL_MEMORY_LINKEDLIST_H
#define TRANSACTIONAL_MEMORY_LINKEDLIST_H

#include <string>
#include "List.h"

class DoublyLinkedList : public List {
public:
  DoublyLinkedList() : List(){};

  void setHead(Node *newHead) {
    head = newHead;
    newHead->previous = nullptr;
  }

  static void setTail(Node *newTail) { newTail->next = nullptr; }

  void insertNode(/* with */ int content, /* afterNodeWith */ int immediatelyPrecedingContent) const;
  void appendNode(/* with */ int content);
  void deleteFirstNode(/* with */ int content);
};

#endif // TRANSACTIONAL_MEMORY_LINKEDLIST_H
