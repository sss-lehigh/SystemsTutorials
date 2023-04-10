// Influenced by bhttps://github.com/aozturk

#include <chrono>
#include <cstddef>
#include <thread>

#include "rome/logging/logging.h"
#include "rome/rdma/memory_pool/memory_pool.h"
#include "util.h"

template <typename K, typename V>
class Node {
public:
    Node(const K &key, const V &value) :
        key(key), value(value), next(NULL) {
    }

    K getKey() const {
        return key;
    }

    V getValue() const {
        return value;
    }

    void setValue(V value) {
        Node::value = value;
    }

    Node *getNext() const {
        return next;
    }

    void setNext(Node *next) {
        Node::next = next;
    }

private:
    // key-value pair
    K key;
    V value;
    // next bucket with the same key
    Node *next;
};

// Default hash function
template <typename K>
struct HashFunc {
    unsigned long operator()(const K& key) const
    {
        return reinterpret_cast<unsigned long>(key) % TABLE_SIZE;
    }
};

template <typename K, typename V, typename F = HashFunc<K> >
class KVStore {
public:

    KVStore(){
        // construct zero initialized hash kv_ of size
        kv_ = new KVStore<K, V> *[TABLE_SIZE]();
    }

    ~KVStore() {
        // destroy all buckets one by one
        for (int i = 0; i < TABLE_SIZE; ++i) {
            Node<K, V> *entry = kv_[i];
            while (entry != NULL) {
                Node<K, V> *prev = entry;
                entry = entry->getNext();
                delete prev;
            }
            kv_[i] = NULL;
        }
        // destroy the hash kv_
        delete [] kv_;
    }

    bool get(const K &key, V &value) {
        unsigned long hashValue = hash_func_(key);
        Node<K, V> *entry = kv_[hashValue];

        while (entry != NULL) {
            if (entry->getKey() == key) {
                value = entry->getValue();
                return true;
            }
            entry = entry->getNext();
        }
        return false;
    }

    void put(const K &key, const V &value) {
        unsigned long hashValue = hash_func_(key);
        Node<K, V> *prev = NULL;
        Node<K, V> *entry = kv_[hashValue];

        while (entry != NULL && entry->getKey() != key) {
            prev = entry;
            entry = entry->getNext();
        }

        if (entry == NULL) {
            entry = new Node<K, V>(key, value);
            if (prev == NULL) {
                // insert as first bucket
                kv_[hashValue] = entry;
            } else {
                prev->setNext(entry);
            }
        } else {
            // just update the value
            entry->setValue(value);
        }
    }

    void remove(const K &key) {
        unsigned long hashValue = hash_func_(key);
        Node<K, V> *prev = NULL;
        Node<K, V> *entry = kv_[hashValue];

        while (entry != NULL && entry->getKey() != key) {
            prev = entry;
            entry = entry->getNext();
        }

        if (entry == NULL) {
            // key not found
            return;
        }
        else {
            if (prev == NULL) {
                // remove first bucket of the list
                kv_[hashValue] = entry->getNext();
            } else {
                prev->setNext(entry->getNext());
            }
            delete entry;
        }
    }

private:
    // hash kv_
    KVStore<K, V> **kv_;
    F hash_func_;
};