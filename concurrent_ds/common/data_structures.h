/* 
 * File:   data_structures.h
 * Author: Olivia Grimes
 */

#ifndef DATA_STRUCTURES_H
#define DATA_STRUCTURES_H

#if defined SEQUENTIAL
#include "../bst_variations/seq_bst/seq_bst.h"
#elif defined HOH_LOCKING
#include "../bst_variations/hoh_bst/hoh_bst.h"
#elif defined OPTIMISTIC_LOCKING
#include "../bst_variations/optimistic_locking/opt_bst.h"
#elif defined SEQUENTIAL_TUTORIAL
#include "../bst_tutorial/seq_bst/seq_bst.h"
#elif defined HOH_LOCKING_TUTORIAL
#include "../bst_tutorial/hoh_bst/hoh_bst.h"
#elif defined OPTIMISTIC_LOCKING_TUTORIAL
#include "../bst_tutorial/optimistic_locking/opt_bst.h"
#else
#error "NO FILE"
#endif

#endif /* DATA_STRUCTURES_H */

