#ifndef TRANSACTIONAL_MEMORY_LIST_H
#define TRANSACTIONAL_MEMORY_LIST_H

#include <cassert>
#include <iostream>
#include <string>
#include "Node.h"

class List {
  friend class DoublyLinkedList;
  friend class ConcurrentDoublyLinkedList;

  Node *head;
  [[nodiscard]] Node *getTail() const {
    if (head == nullptr) { return nullptr; }
    auto currentNode = head;
    while (!currentNode->isTail()) {
      currentNode = currentNode->next;
    }
    return currentNode;
  }

public:
  List() = default;

public:
  class Operation {
  public:
    enum BackingStorage { Insert, Append, Delete };

  private:
    BackingStorage backingStorage;

  public:
    Operation() = default;
    constexpr Operation(BackingStorage backingStorage)
        : backingStorage(backingStorage) {}
    // Allow switching over and comparing an instance using the backing enum.
    constexpr operator BackingStorage() const { return backingStorage; }
    // Disallow treating it as a Boolean value.
    explicit operator bool() const = delete;

  public:
    [[nodiscard]] constexpr std::string_view description() const {
      switch (backingStorage) {
      case Insert:
        return "insert";
      case Append:
        return "append";
      case Delete:
        return "delete";
      }
    }
  };

private:
  class OperationStatus {
  public:
    enum BackingStorage { InProgress, Success, Failure };

  private:
    BackingStorage backingStorage;

  public:
    OperationStatus() = default;
    constexpr OperationStatus(BackingStorage backingStorage)
        : backingStorage(backingStorage) {}
    // Allow switching over and comparing an instance using the backing enum.
    constexpr operator BackingStorage() const { return backingStorage; }
    // Disallow treating it as a Boolean value.
    explicit operator bool() const = delete;
  };

public:
  [[nodiscard]] std::string description() const {
    std::string partialDescription;
    auto currentNode = head;
    if (currentNode != nullptr) {
      while (!currentNode->isTail()) {
        partialDescription += std::to_string(currentNode->content) + " ⇄ ";
        currentNode = currentNode->next;
      }
      partialDescription += std::to_string(currentNode->content);
    }
    return partialDescription;
  }

  void report(OperationStatus status,
              /* of */ Operation operation,
              /* with */ int content,
              int immediatelyPrecedingContent = 0) const {
    switch (status) {
    case OperationStatus::InProgress:
      std::cout << "attempting to " << operation.description()
                << " " << (operation == Operation::Delete ? "first" : "") << " node with content '" << content << "'";
      if (operation == Operation::Insert) {
        std::cout << " immediately after that with content '" << immediatelyPrecedingContent << "'";
      }
      std::cout << std::endl;
      return;
    case OperationStatus::Success:
      std::cout << operation.description()
                << (operation == Operation::Delete ? "d first" : "ed") << " node with content '" << content << "'";
      if (operation == Operation::Append) {
        std::cout << " to end of list";
      } else if (operation == Operation::Insert) {
        std::cout << " after node with content '" << immediatelyPrecedingContent << "'";
      }
      std::cout << std::endl;
      break;
    case OperationStatus::Failure:
      assert(operation != Operation::Append && "cannot fail to append node");
      std::cout << "no node with content '" << (operation == Operation::Delete ? content : immediatelyPrecedingContent)
                << "' to " << operation.description();
      if (operation == Operation::Insert) {
        std::cout << " after";
      }
      std::cout << std::endl;
      break;
    }

    std::cout << "current list: " << description() << std::endl << std::endl;
  }
};

#endif // TRANSACTIONAL_MEMORY_LIST_H
