# Concurrent Data Structures Tutorial

(see here: https://gist.github.com/DomPizzie/7a5ff55ffa9081f2de27c315f5018afc)
This repo is meant to serve as a tutorial to learn how to implement concurrent data structures.

## Description

TODO: An in-depth paragraph about your project and overview of use.

## Getting Started

### Installing

* Clone this repo into your sunlab account: login to your sunlab account on the terminal, and in the directory of your choice, run the following command:
```
git clone https://github.com/sss-lehigh/SystemsTutorials.git
```

* You will additionally need to mount a remote filesystem (i.e, sunlab) to your IDE of choice. Please see instructions here

### Executing program

TODO:
* How to run the program
* Step-by-step bullets
```
code blocks for commands
```

## Tutorial

Before beginning this tutorial, please follow the installation instructions above.

### Sequential BST Implementation

### Hand over hand locking

1. Open node.h and add a mutex field. This enables fine-grained locking at the node-level via mutexes:
```c++
public:
    int key;
    nodeptr left;
    nodeptr right;
    mutex mtx;
```

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

## Version History

* 0.2
    * Various bug fixes and optimizations
    * See [commit change]() or See [release history]()
* 0.1
    * Initial Release

## License

This project is licensed under the [NAME HERE] License - see the LICENSE.md file for details

## Acknowledgments

Inspiration, code snippets, etc.
* [awesome-readme](https://github.com/matiassingers/awesome-readme)
* [PurpleBooth](https://gist.github.com/PurpleBooth/109311bb0361f32d87a2)
* [dbader](https://github.com/dbader/readme-template)
* [zenorocha](https://gist.github.com/zenorocha/4526327)
* [fvcproductions](https://gist.github.com/fvcproductions/1bfc2d4aecb01a834b46)