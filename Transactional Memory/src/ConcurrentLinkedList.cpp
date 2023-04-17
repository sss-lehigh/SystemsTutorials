#include "ConcurrentLinkedList.h"

void ConcurrentSortedDoublyLinkedList::insertNode(int key) {
  report(OperationStatus::InProgress, Operation::Insert, key);

  if (containsNode(key).has_value()) {
    report(OperationStatus::Failure, Operation::Insert, key);
    return;
  }

  auto newNode = new Node(key);

  if (isEmpty()) {
    head = newNode;
    report(OperationStatus::Success, Operation::Insert, key);
  } else {
    auto currentNode = head;
    while (!currentNode->isTail() && currentNode->next->key < key) {
      currentNode = currentNode->next;
    }

    if (!currentNode->isTail()) { newNode->setNext(currentNode->next); }
    newNode->setPrevious(currentNode);
  }

  report(OperationStatus::Success, Operation::Insert, key);
}

void ConcurrentSortedDoublyLinkedList::deleteNode(int key) {
  report(OperationStatus::InProgress, Operation::Delete, key);

  auto optionalNode = containsNode(key);
  if (!optionalNode.has_value()) {
    report(OperationStatus::Failure, Operation::Delete, key);
  } else {
    auto node = optionalNode.value();
    if (node->isHead()) {
      setHead(node->next);
    } else if (node->isTail()) {
      setTail(node->previous);
    } else {
      node->previous->setNext(node->next);
    }
    delete node;

    report(OperationStatus::Success, Operation::Delete, key);
  }
}

std::optional<Node *> ConcurrentSortedDoublyLinkedList::containsNode(int key) const {
  if (isEmpty()) { return std::nullopt; }

  auto currentNode = head;
  while (!currentNode->isTail() && currentNode->key < key) {
    currentNode = currentNode->next;
  }

  if (currentNode->key == key) {
    return currentNode;
  } else {
    return std::nullopt;
  }
}
