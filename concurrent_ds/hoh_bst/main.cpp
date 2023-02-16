/* 
 * File:   main.cpp
 * Author: Olivia Grimes
 *
 * Created on February 6, 2023
 */
#include <vector>
#include <iostream>
#include <unistd.h>
#include <atomic>
#include <chrono>
#include <thread>
#include "bst.h"

#ifndef OPS_BETWEEN_TIME_CHECKS
#define OPS_BETWEEN_TIME_CHECKS 500
#endif

using namespace std;

// user options
struct {
    // variables passed by user
    int seconds;
    int threads;
    int inserts;
    int removes;
    int contains;
    int size;
} cfg;

struct {
    vector<int> num_contains;
    vector<int> num_insert;
    vector<int> num_remove;

    chrono::time_point<chrono::high_resolution_clock> startTime;
    chrono::time_point<chrono::high_resolution_clock> endTime;

    long elapsedMillis;
    long elapsedMillisNapping;

    // whether timing has started
    bool start;
    bool done;

    // number of threads running
    atomic<int> running;
} info;

// the tree
BST* tree;

void insert(BST* tree, int key, int tid) {
    printf("Inserting: %i\n", key);
    bool ret = tree->insert(key);
    if (!ret) {
        printf("   Key %i already inserted.\n", key);
    } else {
        info.num_insert.at(tid) += 1;
        tree->printInOrder();
    }
    printf("\n");
}

void remove(BST* tree, int key, int tid) {
    printf("\nRemoving: %i\n", key);
    bool ret = tree->remove(key);
    if (!ret) {
        printf("   Key %i not present in structure.\n", key);
    } else {
        info.num_remove.at(tid) += 1;
        tree->printInOrder();
    }
    printf("\n");
}

void contains(BST* tree, int key, int tid) {
    printf("\nSearching for: %i\n", key);
    bool present = tree->contains(key);
    if (present)
        printf("  Key %i is present", key);
    else
        printf("  Key %i is NOT present", key);
    info.num_contains.at(tid) += 1;
}

// TODO: implement this
void prefill() {

    // Providing a seed value
	//srand((unsigned) time(NULL));

    int count = 0;
    while (count < 10) {
        count++;
        // Get a random number
	    int random = 1 + (rand() % 100);

        // Print the random number
	    cout<<random<<endl;
    }
}

void *thread_timed(void *_id) {
    int tid = *((int *)_id);
    info.running.fetch_add(1);

    __sync_synchronize();
    while (!info.start) {
        __sync_synchronize();
    }  // wait to start

    int cnt = 0;
    while (!info.done) {
        if (((++cnt) % OPS_BETWEEN_TIME_CHECKS) == 0) {
            chrono::time_point<chrono::high_resolution_clock> __endTime = chrono::high_resolution_clock::now();
            if (chrono::duration_cast<chrono::milliseconds>(__endTime - info.startTime).count() >= abs(cfg.seconds)) {
                __sync_synchronize();
                info.done = true;
                __sync_synchronize();
                break;
            }
        }

        // TODO: DO WORK, i.e., perform operations per desired workload

    }
    pthread_exit(NULL);
}

void trial() {
    info.start = false;
    info.done = false;
    info.elapsedMillis = 0;
    info.elapsedMillisNapping = 0;
    info.running = 0;
    tree = new BST(); // initialize the tree

    // thread data
    pthread_t *threads[cfg.threads];
    int ids[cfg.threads];
    for (int i = 0; i < cfg.threads; ++i) {
        threads[i] = new pthread_t;
        ids[i] = i;
    }

    // TODO: prefill DS
    prefill();

    // start all threads. All worker threads are scheduled first, then range query
    // threads.
    for (int i = 0; i < cfg.threads; ++i) {
        if (pthread_create(threads[i], NULL, thread_timed, &ids[i])) {
            cerr << "ERROR: could not create thread" << endl;
            exit(-1);
        }
    }

    while (info.running.load() < cfg.threads) {
        cout << "main thread: waiting for threads to START running=" << info.running.load() << endl;
    }  // wait for all threads to be ready
    cout << "main thread: starting timer..." << endl;

    cout << endl;
    cout << "###############################################################################" << endl;
    cout << "################################ BEGIN RUNNING ################################" << endl;
    cout << "###############################################################################" << endl;
    cout << endl;

    info.startTime = chrono::high_resolution_clock::now();
    __sync_synchronize();
    info.start = true;

    for (int i = 0; i < cfg.threads; ++i) {
        cout << "joining thread " << i << endl;
        if (pthread_join(*(threads[i]), NULL)) {
            cerr << "ERROR: could not join thread" << endl;
            exit(-1);
        }
    }

    cout << endl;
    cout << "###############################################################################" << endl;
    cout << "################################ END RUNNING ##################################" << endl;
    cout << "###############################################################################" << endl;
    cout << endl;

    for (int i = 0; i < cfg.threads; ++i) {
        delete threads[i];
    }
}

void dummy_trial(int tid) {
    insert(tree, 50, tid);
    insert(tree, 40, tid);
    insert(tree, 60, tid);
    insert(tree, 60, tid);
    insert(tree, 70, tid);
    insert(tree, 10, tid);
    insert(tree, 80, tid);
    insert(tree, 20, tid);
    insert(tree, 55, tid);
    
    printf("\nLevel Order:\n");
    tree->printLevelOrder();

    contains(tree, 60, tid);
    contains(tree, 57, tid);

    remove(tree, 60, tid);
    contains(tree, 60, tid);
    remove(tree, 45, tid);
    
    printf("\nLevel Order:\n");
    tree->printLevelOrder();
}

void usage() {
    std::cout
        << "Command-Line Options:" << std::endl
        << "  -t <int>    : the number of threads in the experiment" << std::endl
        << "  -s <int>    : the number of seconds to perform the experiment" << std::endl
        << "  -h          : display this message and exit" << std::endl;
    exit(0);
}

bool parseargs(int argc, char** argv) {
    // parse the command-line options
    int opt;
    while ((opt = getopt(argc, argv, "t:s:i:r:c:n:h")) != -1) {
        switch (opt) {
          case 't': cfg.threads = atoi(optarg); break;
          case 's': cfg.seconds = atoi(optarg); break;
          case 'i': cfg.inserts = atoi(optarg); break;
          case 'r': cfg.removes = atoi(optarg); break;
          case 'c': cfg.contains = atoi(optarg); break;
          case 'n': cfg.size = atoi(optarg); break;
          case 'h': usage(); exit(0); break;
          default: return false; break;
        }
    }
    return true;
}

bool validateArgs() {
    if (cfg.threads < 0 || cfg.threads > 192) {
        std::cout << "-t must be passed, and be between 0 and 192. Try again.\n";
        return false;
    }
    if (cfg.seconds < 0) {
        std::cout << "Time passed (-s) must be greater than 0. Try again.\n";
        return false;
    }
    // if (cfg.inserts + cfg.contains + cfg.removes != 100) {
    //     std::cout << "The workload parameters passed (-i, -r, -c) must add to 100. Try again.\n";
    //     return false;
    // }
    return true;
}

int main(int argc, char *argv[]) {
    cfg.threads = -1;
    if (!parseargs(argc, argv)) {
        std::cout << "Error parsing args.\n";
        return 0;
    }
    if (!validateArgs()) {
        std::cout << "Error validating args.\n";
        return 0;
    }
    
    for (int i = 0; i < cfg.threads; i++) {
        info.num_insert.push_back(0);
        info.num_remove.push_back(0);
        info.num_contains.push_back(0);
    }
 
    prefill();
    std::this_thread::sleep_for(std::chrono::seconds(2));
    cout << endl << "new one" << endl;
    prefill();
    //trial();

    //dummy_trial(0);

    int num_inserts = 0;
    int num_removes = 0;
    int num_contains = 0;

    for (int i = 0; i < cfg.threads; i++) {
        num_inserts += info.num_insert.at(i);
        num_removes += info.num_remove.at(i);
        num_contains += info.num_contains.at(i);
    }

    int tot_thpt = num_inserts + num_removes + num_contains;

    cout << "Insertion throughput: " << num_inserts << endl;
    cout << "Deletion throughput: " << num_removes << endl;
    cout << "Contains throughput: " << num_contains << endl;
    cout << "Total throughput: " << tot_thpt << endl;
}