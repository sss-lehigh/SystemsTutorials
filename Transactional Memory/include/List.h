#ifndef TRANSACTIONAL_MEMORY_LIST_H
#define TRANSACTIONAL_MEMORY_LIST_H

#include <cassert>
#include <iostream>
#include <string>
#include <optional> // std::optional
#include "Node.h"

class List {
  friend class SortedDoublyLinkedList;
  friend class ConcurrentSortedDoublyLinkedList;

  Node *head;
  [[nodiscard]] Node *getTail() const {
    if (head == nullptr) {
      return nullptr;
    }
    auto currentNode = head;
    while (!currentNode->isTail()) {
      currentNode = currentNode->next;
    }
    return currentNode;
  }

public:
  List() = default;

public:
  virtual void insertNode(/* with */ int key) {
    assert(false && "this method should be overridden by a subclass");
  }
  virtual void deleteNode(/* with */ int key) {
    assert(false && "this method should be overridden by a subclass");
  }
  [[nodiscard]] virtual std::optional<Node*> containsNode(/* with */ int key) const {
    assert(false && "this method should be overridden by a subclass");
  }

  [[nodiscard]] virtual bool isEmpty() const { return head == nullptr; }

public:
  class Operation {
  public:
    enum BackingStorage { Insert, Delete };

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
      case Insert: return "insert";
      case Delete: return "delete";
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
        partialDescription += std::to_string(currentNode->key) + " ⇄ ";
        currentNode = currentNode->next;
      }
      partialDescription += std::to_string(currentNode->key);
    }
    return partialDescription;
  }

  void report(OperationStatus status,
              /* of */ Operation operation,
              /* onNodeWith */ int key) const {
    switch (status) {
    case OperationStatus::InProgress:
      std::cout << "attempting to " << operation.description()
                << " node with key '" << key << "'"
                << std::endl; return;
    case OperationStatus::Success:
      std::cout << operation.description()
                << (operation == Operation::Delete ? "d" : "ed")
                << " node with key '" << key << "'"
                << std::endl; break;
    case OperationStatus::Failure:
      std::cout << "node with key '" << key << "' ";
      switch (operation) {
      case Operation::Insert: std::cout << "already exists"; break;
      case Operation::Delete: std::cout << "not found";      break;
      }
      std::cout << std::endl; break;
    }

    std::cout << "current list: " << description() << std::endl << std::endl;
  }
};

#endif // TRANSACTIONAL_MEMORY_LIST_H
