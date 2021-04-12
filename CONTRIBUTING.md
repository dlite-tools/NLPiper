# Contributing

When contributing to this repository, please first discuss the change you wish to make via issue,
email, or any other method with the maintainers of this repository before making a change.

Please note we have a code of conduct, please follow it in all your interactions with the project.

## Contributing to the Codebase

The code is hosted on [GitHub](https://github.com/tomassosorio/NLPiper),
so you will need to use [Git](http://git-scm.com/) to clone the project and make
changes to the codebase. Once you have obtained a copy of the code, you should
create a development environment that is separate from your existing Python
environment so that you can make and test changes without compromising your
own work environment.


### Creating a Python environment

To create an isolated development environment:

* Install [Poetry](https://python-poetry.org/)
* Make sure that you have [cloned the repository](https://github.com/tomassosorio/NLPiper)
* `cd` to the NLPiper source directory
* Build environment. Run `poetry install`


### Run the test suite locally

Before submitting your changes for review, make sure to check that your changes
do not break any tests by running:

```
make all
```

### Pull Request Process

1. Ensure any install or build dependencies are removed before the end of the layer when doing a
   build.
2. Update the [README.md](README) with details of changes to the interface, this includes new environment
   variables, exposed ports, useful file locations and container parameters.
3. Increase the version numbers in any examples files and the [README.md](README) to the new version that this
   Pull Request would represent. The versioning scheme we use is [semantic versioning](http://semver.org/).


## Code of Conduct

### Our Pledge

In the interest of fostering an open and welcoming environment, we as
contributors and maintainers pledge to making participation in our project and
our community a harassment-free experience for everyone, regardless of age, body
size, disability, ethnicity, gender identity and expression, level of experience,
nationality, personal appearance, race, religion, or sexual identity and
orientation.

### Our Standards

Examples of behavior that contributes to creating a positive environment
include:

* Using welcoming and inclusive language
* Being respectful of differing viewpoints and experiences
* Gracefully accepting constructive criticism
* Focusing on what is best for the community
* Showing empathy towards other community members

Examples of unacceptable behavior by participants include:

* The use of sexualized language or imagery and unwelcome sexual attention or
advances
* Trolling, insulting/derogatory comments, and personal or political attacks
* Public or private harassment
* Publishing others' private information, such as a physical or electronic
  address, without explicit permission
* Other conduct which could reasonably be considered inappropriate in a
  professional setting

### Our Responsibilities

Project maintainers are responsible for clarifying the standards of acceptable
behavior and are expected to take appropriate and fair corrective action in
response to any instances of unacceptable behavior.

Project maintainers have the right and responsibility to remove, edit, or
reject comments, commits, code, wiki edits, issues, and other contributions
that are not aligned to this Code of Conduct, or to ban temporarily or
permanently any contributor for other behaviors that they deem inappropriate,
threatening, offensive, or harmful.