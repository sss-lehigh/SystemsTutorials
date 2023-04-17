#ifndef TRANSACTIONAL_MEMORY_ARGUMENTPARSER_H
#define TRANSACTIONAL_MEMORY_ARGUMENTPARSER_H

#include <algorithm> // std::max
#include <getopt.h>
#include <iostream>
#include <string>
#include <thread>

static unsigned int slowDown = 0;
static unsigned int threadCount = std::max(std::thread::hardware_concurrency(), 2u) - 1;

void parse(const char *arguments[], int argumentCount) {
  static struct option long_options[] = {
      {"help", no_argument, nullptr, 'h'},
      {"slow-down", required_argument, nullptr, 's'},
      {"thread-count", required_argument, nullptr, 't'},
      {nullptr, 0, nullptr, 0}};

  while (true) {
    int option_index = 0;
    int c = getopt_long(argumentCount, const_cast<char **>(arguments), "s:t:", long_options, &option_index);
    if (c == -1) { break; }

    switch (c) {
    case 'h':
      std::cout << "usage: " << arguments[0] << " [options]" << std::endl;
      std::cout << "options:" << std::endl;
      std::cout << "  -h, --help" << std::endl;
      std::cout << "  -s, --slow-down=milliseconds"
                << " (default: " << slowDown << ")" << std::endl;
      std::cout << "  -t, --thread-count=number"
                << " (default: " << threadCount << ")" << std::endl;
      std::exit(EXIT_SUCCESS);
    case 's':
      slowDown = std::stoi(optarg);
      break;
    case 't':
      threadCount = std::stoi(optarg);
      break;
    default:
      std::cerr << "unrecognized option: " << c << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }

  std::cout << "thread count: " << threadCount << std::endl;
};

#endif // TRANSACTIONAL_MEMORY_ARGUMENTPARSER_H
