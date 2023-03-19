# Concurrent Data Structures Tutorial

(see here: https://gist.github.com/DomPizzie/7a5ff55ffa9081f2de27c315f5018afc)
This repo is meant to serve as a tutorial to learn how to implement concurrent data structures.

## Description

Use this repo to learn how to transform a sequential data structure to a multi-threaded, concurrent one using three different techniques: (1) hand-over-hand locking, (2) optimistic locking, and (3) a lock-free technique.

## Getting Started

### Important Directories

This tutorial takes place in its entirety in the `concurrent_ds/` directory.

Within the `concurrent_ds/` directory, there is a directory called `bst_tutorial/` and a directory called `bst_solution/`. `bst_solution/` contains the solution for the binary search tree variants that will be implemented in this tutorial, whereas `bst_tutorial/` initially contains the skeleton code, to be updated and implemented in this tutorial.

`common/` contains files required to run the trials, and do not need to be modified in this tutorial.


### Installing

* Clone this repo on to your local computer, in a directory of your choosing:
```
git clone https://github.com/sss-lehigh/SystemsTutorials.git
```

### Building the data structures

1. Send the code to your sunlab account using the `scp` command. From the root directory of this repo (`SystemsTutorials/`), run:
    ```
    scp concurrent_ds/ <sunlab_username>@sunlab.cse.lehigh.edu:~
    ```
    ***NOTE:*** you can choose to put your code in whichever directory you would like on your sublab account (i.e., doesn't have to be your home directory as in the command).

2. Login to your sunlab account, and navigate to the `concurrent_ds/` directory.

    To create the executable for the solution files provided, run:

    ```
    make solution
    ```

    To create the executable for the tutorial files, run:
    ```
    make tutorial
    ```

    You can also choose to make individual executables: by passing at least of the following to `make`:
    * `seq` : sequential BST solution
    * `hoh` : hand-over-hand locking BST solution
    * `opt` : optimistic locking BST solution
    * `seq_tutorial` : sequential BST tutorial
    * `hoh_tutorial` : hand-over-hand locking BST tutorial
    * `opt_tutorial` : optimistic locking BST tutorial

    The executables produced by running the aforementioned `make` commands are named the same as the option passed to `make` to create the executable (i.e., the list right above this).

### Executing the programs

After building executables, you can run them by passing the following options:
* `s` : the time in milliseconds to run the trial
* `k` : the key range, [0, k)
* `i` : the ratio of inserts as a percentage
* `r` : the ratio of removes as a percentage
* `c` : the ratio of contains as a percentage
* `t` : the number of threads to use

For example:
```
./opt -s 3000 -k 1000000 -i 10 -r 10 -c 80 -t 8
```

***NOTE:*** the values passed to `-i`, `-r`, and `-c` must add to 100.

## Tutorial

Before beginning this tutorial, please follow the installation instructions above, and open the `concurrent_ds/` directory in an IDE. Please also familiarize yourself with how to build and execute the programs, which will be needed later on.

### Pre-Tutorial Reading: Sequential BST Implementation

The sequential implementation of a binary search tree which supports adding, removing, and searching for a key, has been provided in the `bst_tutorial/` directory. Let's begin by understanding this implementation.

1. In your IDE, open the three files within the `concurrent_ds/bst_tutorial/` directory (`seq_bst.cpp`, `seq_bst.h`, `seq_node.h`).

2. Let's begin with `seq_node.h`. This file defines the layout of a node in the binary search tree: each node contains a key and a left and a right pointer. There is a constructor which takes in a key, and sets the left and right pointers to NULL. This is a very typical, simple node layout for a node of a binary search tree. Additionally, on line 14, `nodeptr` is defined as a pointer to a node.

3. Now let's move on to `seq_bst.h`. This file defines the binary search tree. It stores a pointer to the root of the tree, and includes the following methods: a constuctor which sets the root node as NULL, methods to insert, remove, and search for a key in the BST. Note that it includes the node header class in order to use the node which we previously defined.

4. Finally, let's move on to the bulk of the implementation, `seq_bst.cpp`, which implements the methods defined in `seq_bst.h`. Examine each of the following methods: `contains()`, `insert()`, and `remove()`, making sure to understand each. Use the comments to help in your understanding. Once you understand these methods, you are ready to begin creating a concurrent version!

### Tutorial 1: Hand-Over-Hand Locking

In this tutorial, we will use a technique called hand-over-hand (HoH) locking to transform the sequential binary search tree into a concurrent one. If you are unfamilar with the concept of HoH locking, please take a moment to read and understand slides 16-30 from the following source: https://courses.csail.mit.edu/6.852/08/lectures/Lecture21.pdf 

This tutorial will edit code in the `concurrent_ds/bst_tutorial/hoh_bst` directory. The code that is currently there is the sequential version of the binary search tree, which we previously spent time understanding.

We are now ready to make modifications to the code.

1. Since we now need the ability to lock and unlock nodes, we must add a mutex field to the definition of a node. This enables fine-grained locking at the node-level via mutexes. Open `hoh_node.h` and change the definition to include a mutex called mtx:

```c++
public:
    int key;
    nodeptr left;
    nodeptr right;
    mutex mtx;
```

2. Briefly open `hoh_bst.h`. You'll note that the only difference from the sequential definition is the definitions `SENTINEL` and `SENTINEL_BEG`. While this is not a requirement for concurrent data structures, in this implementation we will designate sentinel nodes, which are essentially "dummy nodes" at the beginning and end of the data structure. This means that there will be one `SENTINEL_BEG` for the whole tree, and two `SENTINEL`s for each leaf node on the tree. Note that the `SENTINEL_BEG` node's left pointer will always be NULL, and the right pointer will point to what will be referred to as the "true root" of the tree.

3. Now, open `hoh_bst.cpp`, which is where most, and the remainder, of the changes will occur. First, let's modify the constructor (line 9). When a new tree is created, we no longer want to simply set the root to NULL, so you can remove that line. Instead, we want to 




--> notes:
2. Add in sentinel node to the beginning (I chose to make the left pointer NULL and right pointer point to the "true root" as I will refer to it.)

3. Modify the insert method to support concurrently inserting when the root is NULL. Add insertRoot()

3. Add locking and unlocking of parent and curr

4. Create addSentintels() method

### Optimistic locking

1. First create method `verify_traversal(nodeptr prev, nodeptr curr, bool left)`

2. Go through each method and remove HoH mutex calls -- to --> only lock prev and curr and then call verify_traversal()


## Help

Any advise for common problems or issues.
```
command to run if program contains helper info
```

## Authors

Contributors names and contact info

ex. Dominique Pizzie
ex. [@DomPizzie](https://twitter.com/dompizzie)