# Transaction Memory tutorial

Transactional Memory is a programming technique that "magically" turns non-concurrent code concurrent, with little effort from the programmer themselves. In this tutorial, we will do a little programming on a linked list in C++, and see how one can benefit from this technique. 

## Project overview

First, let's download the Docker image [here](link-pending), which contains everything we need, then navigate to the project root `~/SystemTutorials/Transactional Memory/`.

> **Note**
> If you have a release version of Clang 17 or newer, or any C++ compiler with the feature test macro `__cpp_transactional_memory` evaluated to `202110`, then you can skip the Docker image download and work directly with this repository. However, it's still recommended that you use the Docker image, in order to avoid missing dependency problems.

The tutorial is organized as a project with 3 executables. It includes a few files. We'll only focus on 7 of them:

```
project root
  ├ include
  │   ├ LinkedList.h
  │   └ ConcurrentLinkedList.h
  ├ src
  │   ├ LinkedList.cpp
  │   └ ConcurrentLinkedList.cpp
  ├ linked-list-main.cpp
  ├ concurrent-linked-list-main.cpp
  └ transactional-memory-concurrent-linked-list-main.cpp
```

The other files we can ignore, some of which are just headers that abstract away the bloat, so that we can focus on the important parts.

Before we jump into what's in these files, we need to generate the build configurations using CMake:

```
; cmake -G "Ninja" # ";" stands in for a safe-to-copy command prompt
```

We'll then use Ninja to build the 3 executables in the following steps. Each step focuses on 1 executable, with step 2 being potential the most time-consuming (about 20 - 30 minutes).

## Step 1: linked list

There is no coding required in this step, and we'll only be looking at `include/LinkedList.h`, `src/LinkedList.cpp`, and `linked-list-main.cpp`. The main objective of this step is familiarizing ourselves with the project's structure and build process.

Let's first take a look at `linked-list-main.cpp`. This file has a `main` function that serves as the entry point to the `linked-list` executable. Inside `main`, we can see that it runs an infinite loop calling `mutate` on a list in each iteration. The `mutate` function randomly chooses from 2 operations to perform on the list: insert and delete. The details of these 3 operations are defined in `src/LinkedList.cpp`.

To build and run the executable, simply do:

```
; ninja linked-list # build using ninja
; ./linked-list
```

Optionally, we can pass in a `--slow-down` argument (e.g. 500 ms), to better see the operations one by one:

```
; ./linked-list --slow-down=500
```

At the beginning and end of each operation, it announces its status/progress, followed by a description of what the list now looks like.

To stop the executable, send in an interrupt signal (<kbd>⌃C</kbd>).

## Step 2: concurrent linked list

In this step we'll be working with the concurrent counterpart to step 1's files: `include/ConcurrentLinkedList.h`, `src/ConcurrentLinkedList.cpp`, and `concurrent-linked-list-main.cpp`.

Like in step 1, we'll first take a look at the executable's entry point: the `main` function in `concurrent-linked-list-main.cpp`. Notice that the `mutate` function in this file is exactly the same as that in `linked-list-main.cpp`. The only differences are a few `#include` statements and that the `main` function now calls it through `threadCount` number of threads (`threadCount` value can be overridden with the `--thread-count` argument). 

To build and run the executable for this step:

```
; ninja concurrent-linked-list
; ./concurrent-linked-list
```

How did the program run? Did it crash with a segmentation fault, an abort signal, or any error that indicates a bad memory access? If yes, then good, the program behaved as expected; otherwise, set `--thread-count` to an integer > 1 and observe the expected crash:

```
./concurrent-linked-list --thread-count=10
```

> **Note**
> To verify that `concurrent-linked-list` is using multiple threads, you can inspect thread usage via tools such as `top` or its alternatives `htop` and `btop`.

In addition to the program crashing, notice that its output before the crash is all jumbled up, unlike the clean stream we saw in step 1.

To investigate both the crash and the jumbled output, let's take another look at `concurrent-linked-list-main.cpp`. There isn't anything wrong in this file (such as dereferencing a null pointer or undefined behaviors) that would result in a bad memory access. But, notice that right below the `#include` statements, we have a type alias declaration 

```c++
using LinkedList = ConcurrentSortedDoublyLinkedList;
```

different from `linked-list-main.cpp`'s

```c++
using LinkedList = SortedDoublyLinkedList;
```

Maybe there is something in `SortedDoublyLinkedList` causing all the problems, so let's take a look at its declaration in  `include/ConcurrentLinkedList.h`, and its member function implementations in `src/LinkedList.cpp`. As with the `main` function file, compare them with their non-concurrent counterpart `include/LinkedList.h` and `src/LinkedList.cpp`. Notice that except for a few naming differences, `ConcurrentSortedDoublyLinkedList` is practically the same as `SortedDoublyLinkedList`. If we look through the function implementations, we encounter this line in `deleteNode`:

```c++
delete node;
```

Herein lies the root cause of our problems. In the delete operation, we deallocate the `Node` object if found. If another thread happens to be traversing to the object, it will be attempting to read something that doesn't exist anymore (null pointer dereference). On modern systems, instead of letting the program read invalid (and potentially sensitive) data, it traps the program's execution and let it crash. This is fine for our non-concurrent executable, because by being a single-threaded program, there is no conflicting memory access within itself. However, for concurrent programs, access to shared mutable states (e.g. the deleted `Node` object) must be well-synchronized to avoid conflicts. 

Historically, one popular synchronization primitive is lock. It's also one that you should already be familiar with through previous computer science classes. As an exercise, try to implement a lock-based synchronization for `concurrent-linked-list`. Don't spend too much time on it, though; budget about 10 - 15 minutes, at most. 

> **Note**
> In your implementation, try to keep the scope of changes within `include/ConcurrentLinkedList.h` and `src/LinkedList.cpp`. If you find it necessary to modify the node type used in the list, declare a new `Node`-derived type and use it in `List`'s generic parameter. 

Once you're satisfied with your implementation, or if you've run out of your time budget, move on to the next step.

## Step 3

In this step we'll explore how we can use Transactional Memory to achieve synchronization. Before we dive into it, let's briefly review our synchronization effort in step 2.

<details>
    <summary>
        step 2 review
    </summary>
    <p>
        If your synchronization in step 2 worked, good job! If it didn't, it's still fine. Synchronization using locks is difficult, especially performant ones.
    </p>
    <p>
        There are many ways to synchronize access to a linked list using locks: a global lock, hand-over-hand locking on each node, a lock table, etc. Under the `patches/` directory are reference implementations of the first 2 designs. Apply each patch to the repository and check it out.
    </p>
<pre>
; git add .
; git stash # stashes your synchronization implementation
; git am path/to/Transactional\ Memory/patches/global\ lock.patch
</pre>
    <p>
        This first patch implements a global lock. It's the simpliest synchronization but suffers from poor performance because only one thread can work at a time. In fact it's even worse than the non-concurrent `linked-list`, due to the overhead of contending on and aquiring the lock.
    </p>
<pre>
; git restore .
; git clean -f
; git am path/to/Transactional\ Memory/patches/hand-over-hand\ locking.patch
</pre>
    <p>
        This second patch implements a hand-over-hand locking. This is a lot more performant than the global lock, but it also incurs a significantly higher implementation difficulty.
    </p>
</details>

Now, back to Transactional Memory. As with the previous steps, we'll first take a look at the executable's entry file `transactional-memory-concurrent-linked-list-main.cpp`, and compare it against the other 2's. Notice that in this file, like in `concurrent-linked-list-main.cpp`, we are calling `mutate` through multiple threads, but unlike it, we are mutating the non-concurrent linked list like in `linked-list.cpp`. Without making any changes to this file, let's just build and run it and see what happens:

```
; ninja transactional-memory-concurrent-linked-list
; ./transactional-memory-concurrent-linked-list
```

As expected, like in the beginning of step 2, the program crashes due to a bad memory access. 

Instead of adding locks this time, let's uncomment one of both of the `atomic do` keywords in the file. That's it. That's all the change we'll make to this file. Now build and run it again, and see that the non-concurrent linked list is now "magically" concurrency-safe.

This `atomic do` keyword denotes an atomic statement, a new (and as of writing, experimental) C++ feature to provide language-level support to transactional memory. Everything within the scope of an atomic statement is a transaction and guaranteed to be concurrency-safe. Just as with locks, there are many kinds of Transactional Memory implementations. The one we're using in this atomic statement behind the scene is the reference implementation in Clang's libc++abi, which can be swapped out with custom implementations that best fit a given program's workload. This powerful ability supported through such minimal syntax enables performant synchronization through a simplicity comparable to a global lock, giving the programmer the best of the both worlds.
