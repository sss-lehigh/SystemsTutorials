/* 
 * File:   data_structures.h
 * Author: Olivia Grimes
 */

#ifndef DATA_STRUCTURES_H
#define DATA_STRUCTURES_H

#if defined SEQUENTIAL
#include "seq_bst/bst.h"
#elif defined HOH_LOCKING
#include "hoh_bst/bst.h"
#elif defined OPTIMISTIC_LOCKING
#include "optimistic_locking/bst.h"
#else
#include "optimistic_locking/bst.h"
#endif

#endif /* DATA_STRUCTURES_H */

