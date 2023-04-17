#ifndef TRANSACTIONAL_MEMORY_RANDOM_H
#define TRANSACTIONAL_MEMORY_RANDOM_H

#include <random>

std::random_device randomDevice;
std::mt19937 generator(randomDevice());
std::uniform_int_distribution<int> operationRawValueDistribution(0, 2);
std::uniform_int_distribution<int> doubleDigitIntegerDistribution(10, 99);

int randomValue() { return doubleDigitIntegerDistribution(generator); }

#define randomOperation()                                                      \
  LinkedList::Operation(static_cast<List::Operation::BackingStorage>(          \
      operationRawValueDistribution(generator)))

#endif // TRANSACTIONAL_MEMORY_RANDOM_H
