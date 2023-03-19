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
#include <ratio>
#include <thread>
#include <fstream>

#include "data_structures.h"

#include "random.h"
#include "debugprinting.h"

#define SOFTWARE_BARRIER asm volatile("": : :"memory")

#ifndef OPS_BETWEEN_TIME_CHECKS
#define OPS_BETWEEN_TIME_CHECKS 500
#endif

// configure the following for the machine
#define PREFETCH_SIZE_WORDS 24
#define MAX_THREADS 8 
#define MAX_TID_POW2 8

using namespace std;

// user options
double INS;
double DEL;
double RQ;
double CONT;
int RQSIZE;
int MAXKEY;
int MILLIS_TO_RUN;
bool PREFILL;
int WORK_THREADS;
int RQ_THREADS;
int TOTAL_THREADS;

const long long PREFILL_INTERVAL_MILLIS = 100;

// the tree
BST* tree;

struct {
    Random rngs[MAX_TID_POW2 * PREFETCH_SIZE_WORDS];
    
    chrono::time_point<chrono::high_resolution_clock> startTime;
    chrono::time_point<chrono::high_resolution_clock> endTime;

    long elapsedMillis;
    long elapsedMillisNapping;

    // whether timing has started
    bool start;
    bool done;

    // number of threads running
    atomic<int> running;

    vector<int> prefillSize;
} info;

struct {
    vector<int> contains_exists;
    vector<int> contains_dne;
    vector<int> insert_success;
    vector<int> insert_fail;
    vector<int> remove_success;
    vector<int> remove_fail;
} results;

// code run by each thread in trial
void *thread_timed(void *_id) {
    int tid = *((int *)_id);
    Random *rng = &info.rngs[tid * PREFETCH_SIZE_WORDS];
    info.running.fetch_add(1);

    __sync_synchronize();
    while (!info.start) {
        __sync_synchronize();
    }  // wait to start

    int cnt = 0;
    while (!info.done) {
        if (((++cnt) % OPS_BETWEEN_TIME_CHECKS) == 0) {
            chrono::time_point<chrono::high_resolution_clock> __endTime = chrono::high_resolution_clock::now();
            if (chrono::duration_cast<chrono::milliseconds>(__endTime - info.startTime).count() >= abs(MILLIS_TO_RUN)) {
                __sync_synchronize();
                info.done = true;
                __sync_synchronize();
                break;
            }
        }
        
        int key = rng->nextNatural(MAXKEY);
        double op = rng->nextNatural(100000000) / 1000000.;
        // insert
        if (op < INS) {
            if (tree->insert(key)) {
                results.insert_success.at(tid) += 1;
            } else {
                results.insert_fail.at(tid) += 1;
            }
        }
        // delete
        else if (op < INS + DEL) {
            if (tree->remove(key)) {
                results.remove_success.at(tid) += 1;
            } else {
                results.remove_fail.at(tid) += 1;
            }
        }
        // contains
        else {
            if (tree->contains(key)) {
                results.contains_exists.at(tid) += 1;
            } else {
                results.contains_dne.at(tid) += 1;
            }
        }
    }

    info.running.fetch_add(-1);
    while (info.running.load()) { /* wait */ }
    pthread_exit(NULL);
}

// prefill the data structure to 50%
void *thread_prefill(void *_id) {
    int tid = *((int *)_id);
    Random *rng = &info.rngs[tid * PREFETCH_SIZE_WORDS];
    
    double insProbability = 50.0;
    info.running.fetch_add(1);
    while (!info.start) {
        __sync_synchronize();
        TRACE COUTATOMICTID("waiting to start" << endl);
    }  // wait to start

    int cnt = 0;

    while (!info.done) {
        if (((++cnt) % OPS_BETWEEN_TIME_CHECKS) == 0) {
            chrono::time_point<chrono::high_resolution_clock> __endTime = chrono::high_resolution_clock::now();
            if (chrono::duration_cast<chrono::milliseconds>(__endTime - info.startTime).count() >= abs(PREFILL_INTERVAL_MILLIS * ((long long)1000000))) {
                __sync_synchronize();
                info.done = true;
                __sync_synchronize();
                break;
            }
        }

        int key = rng->nextNatural(MAXKEY);
        double op = rng->nextNatural(100000000) / 1000000.;
        if (op < insProbability) {
            if (tree->insert(key)) {
                info.prefillSize.at(tid) += 1;
            }
        } else {
            if (tree->remove(key)) {
                info.prefillSize.at(tid) -= 1;
            }
        }
    }
    info.running.fetch_add(-1);
    while (info.running.load()) { /* wait */ }
    pthread_exit(NULL);
}

void prefill() {
    const int MAX_ATTEMPTS = 1000;
    const double PREFILL_THRESHOLD = 0.01;
    const double expectedFullness = 0.5;  // percent full in expectation
    const int expectedSize = (int)(MAXKEY * expectedFullness);

    int sz = 0;
    int attempts;
    for (attempts = 0; attempts < MAX_ATTEMPTS; ++attempts) {
        // init the DS
        pthread_t *threads[TOTAL_THREADS];
        int *ids = new int[TOTAL_THREADS];
        for (int i = 0; i < TOTAL_THREADS; ++i) {
            threads[i] = new pthread_t;
            ids[i] = i;
        }

        // start all threads
        for (int i = 0; i < TOTAL_THREADS; ++i) {
            if (pthread_create(threads[i], NULL, thread_prefill, &ids[i])) {
                cerr << "ERROR: could not create thread" << endl;
                exit(-1);
            }
        }

        while (info.running.load() < TOTAL_THREADS) {}
        info.startTime = chrono::high_resolution_clock::now();
        __sync_synchronize();
        info.start = true;

        // amount of time for main thread to wait for children threads
        timespec tsExpected;
        tsExpected.tv_sec = 0;
        tsExpected.tv_nsec = PREFILL_INTERVAL_MILLIS * ((long long)1000000);
        // short nap
        timespec tsNap;
        tsNap.tv_sec = 0;
        tsNap.tv_nsec = 10000000;  // 10ms

        nanosleep(&tsExpected, NULL);
        info.done = true;
        __sync_synchronize();

        // wait for threads to complete
        while (info.running.load() > 0) {
            nanosleep(&tsNap, NULL);
        }

        for (int i = 0; i < TOTAL_THREADS; ++i) {
            if (pthread_join(*threads[i], NULL)) {
                cerr << "ERROR: could not join prefilling thread" << endl;
                exit(-1);
            }
        }
        delete[] ids;
        for (int i = 0; i < TOTAL_THREADS; i++) {
            delete threads[i];
        }

        info.start = false;
        info.done = false;
        sz = 0;

        for (int i = 0; i < TOTAL_THREADS; i++) {
            sz += info.prefillSize.at(i);
        }

        if (sz > expectedSize * (1 - PREFILL_THRESHOLD)) {
            break;
        } else {
            cout << "finished attempt ds size: " << sz << endl;
        }
    }

    if (attempts >= MAX_ATTEMPTS) {
        cerr << "ERROR: could not prefill to expected size " << expectedSize
            << ". reached size " << sz << " after " << attempts << " attempts"
            << endl;
        exit(-1);
    }

    COUTATOMIC("finished prefilling to size " << sz << " for expected size " << expectedSize << " " << endl);
}

// perform a trial
void trial() {
    info.start = false;
    info.done = false;
    info.elapsedMillis = 0;
    info.elapsedMillisNapping = 0;
    info.running = 0;
    tree = new BST(); // initialize the tree

    // get random number generator seeded with time
    // we use this rng to seed per-thread rng's that use a different algorithm
    srand(time(NULL));
    
    // thread data
    pthread_t *threads[TOTAL_THREADS];
    int ids[TOTAL_THREADS];
    for (int i = 0; i < TOTAL_THREADS; ++i) {
        threads[i] = new pthread_t;
        ids[i] = i;
        info.rngs[i * PREFETCH_SIZE_WORDS].setSeed(rand());
    }

    prefill();

    // amount of time for main thread to wait for children threads
    timespec tsExpected;
    tsExpected.tv_sec = MILLIS_TO_RUN / 1000;
    tsExpected.tv_nsec = (MILLIS_TO_RUN % 1000) * ((long long)1000000);

    // short nap
    timespec tsNap;
    tsNap.tv_sec = 0;
    tsNap.tv_nsec = 10000000;  // 10ms

    // start all threads
    for (int i = 0; i < TOTAL_THREADS; ++i) {
        if (pthread_create(threads[i], NULL, thread_timed, &ids[i])) {
            cerr << "ERROR: could not create thread" << endl;
            exit(-1);
        }
    }

    while (info.running.load() < TOTAL_THREADS) {
        cout << "main thread: waiting for threads to START running=" << info.running.load() << endl;
    }  // wait for all threads to be ready
    cout << "main thread: starting timer..." << endl;

    cout << endl;
    cout << "###############################################################################" << endl;
    cout << "################################ BEGIN RUNNING ################################" << endl;
    cout << "###############################################################################" << endl;
    cout << endl;

    SOFTWARE_BARRIER;
    info.startTime = chrono::high_resolution_clock::now();
    __sync_synchronize();
    info.start = true;
    SOFTWARE_BARRIER;

    if (MILLIS_TO_RUN > 0) {
        nanosleep(&tsExpected, NULL); // sleep for the duration of the trial
        SOFTWARE_BARRIER;
        info.done = true;
        __sync_synchronize();
    }

    info.elapsedMillis = chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - info.startTime).count();
    info.elapsedMillisNapping = 0;
    while (info.running.load() > 0) {
        nanosleep(&tsNap, NULL);
        info.elapsedMillisNapping = chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - info.startTime).count() - info.elapsedMillis;
    }

    // join the children threads with this main thread
    for (int i = 0; i < TOTAL_THREADS; ++i) {
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

    cout << ((info.elapsedMillis + info.elapsedMillisNapping) / 1000.) << "s" << endl;

    for (int i = 0; i < TOTAL_THREADS; ++i) {
        delete threads[i];
    }
}

// message printed for -h
void usage() {
    std::cout
        << "Command-Line Options:" << std::endl
        << "  -t <int>    : the number of threads in the experiment" << std::endl
        << "  -s <int>    : the number of milliseconds to perform the experiment" << std::endl
        << "  -i <int>    : percentage of inserts (ex. \"-i 10\" is 10 percent inserts)" << std::endl
        << "  -r <int>    : percentage of removes" << std::endl
        << "  -c <int>    : percentage of contains" << std::endl
        << "  -k <int>    : maximum key" << std::endl
        << "  -h          : display this message and exit" << std::endl;
    exit(0);
}

// parse the command-line options
bool parseargs(int argc, char** argv) {
    int opt;
    while ((opt = getopt(argc, argv, "t:s:i:r:c:k:h")) != -1) {
        switch (opt) {
          case 't': TOTAL_THREADS = atoi(optarg); break;
          case 's': MILLIS_TO_RUN = atoi(optarg); break;
          case 'i': INS = atoi(optarg); break;
          case 'r': DEL = atoi(optarg); break;
          case 'c': CONT = atoi(optarg); break;
          case 'k': MAXKEY = atoi(optarg); break;
          case 'h': usage(); exit(0); break;
          default: return false; break;
        }
    }
    return true;
}

// validate arguments passed by the user
bool validateArgs() {
    if (TOTAL_THREADS < 0 || TOTAL_THREADS > MAX_THREADS) {
        std::cout << "-t must be passed, and be between 0 and " << MAX_THREADS << ". Try again.\n";
        return false;
    }
    if (MILLIS_TO_RUN < 0) {
        std::cout << "Time passed (-s) must be greater than 0. Try again.\n";
        return false;
    }
    if (INS + CONT + DEL != 100) {
        std::cout << "The workload parameters passed (-i, -r, -c) must add to 100. Try again.\n";
        return false;
    }
    if (MAXKEY < 10) {
        std::cout << "-k (max key range) must be passed, and be at least 10.\n";
        return false;
    }
   #ifdef SEQUENTIAL
    if (TOTAL_THREADS > 1) {
        std::cout << "Sequential only supports one thread.\n";
        return false;
    }
   #endif
    return true;
}

/////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////
//
// DUMMY TRIAL TO HOPEFULLY FIND BUG
//
/////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////


bool insert(int key) {
    cout << "Attempting to insert key " << key << endl;
    if (tree->insert(key)) {
        cout << "Successfully inserted " << key << endl << endl;
        return true;
    }
    return false;
}

bool remove(int key) {
    cout << "Attempting to remove key " << key << endl;
    if (tree->remove(key)) {
        cout << "Successfully removed " << key << endl << endl;
        return true;
    }
    return false;
}

bool contains(int key) {
    cout << "Searching for key " << key << endl;
    if (tree->contains(key)) {
        cout << "Successfully found " << key << endl << endl;
        return true;
    }
    return false;
}

void workload1() {
    sleep(1);
    insert(647);
}

void workload2() {
    sleep(1);

    
}

void dummyTrial() {
    tree = new BST(); // initialize the tree

}



/////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////
//
// END DUMMY TRIAL
//
/////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////


int main(int argc, char *argv[]) {
    TOTAL_THREADS = -1;
    if (!parseargs(argc, argv)) {
        std::cout << "Error parsing args.\n";
        return 0;
    }
    if (!validateArgs()) {
        std::cout << "Error validating args.\n";
        return 0;
    }
    
    for (int i = 0; i < TOTAL_THREADS; i++) {
        results.insert_success.push_back(0);
        results.insert_fail.push_back(0);
        results.remove_success.push_back(0);
        results.remove_fail.push_back(0);
        results.contains_exists.push_back(0);
        results.contains_dne.push_back(0);
        info.prefillSize.push_back(0);
    }

    trial();

    #ifdef OPTIMISTIC_LOCKING
    tree->validateDS();
    #endif

    // print output
    int num_inserts = 0;
    int num_removes = 0;
    int num_contains = 0;

    for (int i = 0; i < TOTAL_THREADS; i++) {
        num_inserts += results.insert_success.at(i);
        num_removes += results.remove_success.at(i);
        num_contains += results.contains_exists.at(i);
    }

    int tot_thpt = num_inserts + num_removes + num_contains;

    cout << "Insertion throughput: " << num_inserts << endl;
    cout << "Deletion throughput: " << num_removes << endl;
    cout << "Contains throughput: " << num_contains << endl;
    cout << "Total throughput: " << tot_thpt << endl;

    return 0;
}