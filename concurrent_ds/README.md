# Concurrent Data Structures Tutorial

(see here: https://gist.github.com/DomPizzie/7a5ff55ffa9081f2de27c315f5018afc)
This repo is meant to serve as a tutorial to learn how to implement concurrent data structures.

## Description

TODO: An in-depth paragraph about your project and overview of use.

## Getting Started

### Dependencies

TODO:
* Describe any prerequisites, libraries, OS version, etc., needed before installing program.
* ex. Windows 10

### Installing

TODO:
* How/where to download your program
* Any modifications needed to be made to files/folders

### Executing program

TODO:
* How to run the program
* Step-by-step bullets
```
code blocks for commands
```

## Tutorial

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