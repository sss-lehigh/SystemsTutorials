/* 
 * File:   data_structures.h
 * Author: Olivia Grimes
 */

#ifndef DATA_STRUCTURES_H
#define DATA_STRUCTURES_H

#if defined SEQUENTIAL
#include "bst_variations/seq_bst/bst.h"
#elif defined HOH_LOCKING
#include "bst_variations/hoh_bst/bst.h"
#elif defined OPTIMISTIC_LOCKING
#include "bst_variations/optimistic_locking/bst.h"
#else
#include "bst_variations/optimistic_locking/bst.h"
#endif

#endif /* DATA_STRUCTURES_H */

