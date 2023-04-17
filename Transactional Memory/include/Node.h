#ifndef TRANSACTIONAL_MEMORY_NODE_H
#define TRANSACTIONAL_MEMORY_NODE_H

class Node {
public:
  // We use pointers instead of optionals for dramatic effects.
  Node *previous = nullptr;
  Node *next = nullptr;
  int key;

  explicit Node(int content) : key(content) {}

  void setPrevious(Node *newPrevious) {
    previous = newPrevious;
    newPrevious->next = this;
  }

  void setNext(Node *newNext) {
    next = newNext;
    newNext->previous = this;
  }

  [[nodiscard]] bool isHead() const { return previous == nullptr; }
  [[nodiscard]] bool isTail() const { return next == nullptr; }
};

#endif // TRANSACTIONAL_MEMORY_NODE_H
