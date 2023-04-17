#include "LinkedList.h"

void DoublyLinkedList::insertNode(/* with */ int content,
                                  /* afterNodeWith */ int immediatelyPrecedingContent) const {
  report(OperationStatus::InProgress, Operation::Insert, /* content */ content, immediatelyPrecedingContent);
  auto currentNode = head;
  while (currentNode != nullptr && currentNode->content != immediatelyPrecedingContent) {
    currentNode = currentNode->next;
  }
  if (currentNode == nullptr) {
    report(OperationStatus::Failure, Operation::Insert, /* content */ content, immediatelyPrecedingContent);
  } else {
    auto newNode = new Node(content);
    if (!currentNode->isTail()) { newNode->setNext(currentNode->next); }
    newNode->setPrevious(currentNode);
    report(OperationStatus::Success, Operation::Insert, /* content */ newNode->content, /* immediatelyPrecedingContent */ currentNode->content);
  }
}

void DoublyLinkedList::appendNode(/* with */ int content) {
  report(OperationStatus::InProgress, Operation::Append, content);
  auto newNode = new Node(content);
  auto tail = getTail();
  if (tail == nullptr) {
    head = newNode;
  } else {
    tail->setNext(newNode);
  }
  report(OperationStatus::Success, Operation::Append, newNode->content);
}

void DoublyLinkedList::deleteFirstNode(/* with */ int content) {
  report(OperationStatus::InProgress, Operation::Delete, content);
  auto currentNode = head;
  while (currentNode != nullptr && currentNode->content != content) {
    currentNode = currentNode->next;
  }
  if (currentNode == nullptr) {
    report(OperationStatus::Failure, Operation::Delete, content);
  } else {
    if (currentNode->isHead()) {
      setHead(currentNode->next);
    } else if (currentNode->isTail()) {
      setTail(currentNode->previous);
    } else {
      currentNode->previous->setNext(currentNode->next);
    }
    delete currentNode;
    report(OperationStatus::Success, Operation::Delete, content);
  }
}
