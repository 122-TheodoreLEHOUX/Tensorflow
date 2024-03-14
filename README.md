# Document Tensorflow

## General

Creating a TensorFlow.js-Compatible Model and Configuring GPU Usage

## Platform Compatibility

| Platform    | Compatible  | Comment     |
|-------------|-------------|-------------|
| Windows     |      Y      |             |
| MAC         |      N      |             |
| Linux       |      N      |             |

## Tools

The following tools are required:

| Tool          | Version        | Comment      |
|---------------|----------------|--------------|
| Python        | 3.10.2          |              |

## Development Environment Setup and Instructions

1. Install Python

2. Ensure pip is up to date: py -3.10 -m pip install -U pip

3. Ensure the following PATH variables are set
    * C:\Users\XXX\AppData\Local\Programs\Python\PythonXXX\Scripts
    * C:\Users\XXX\AppData\Local\Programs\Python\PythonXXX

4. Run setup.bat to install and setup virtual environment. Do this every time you begin work.

5. Build and install local libraries in the lib folder as required. Follow the respective README.md file install instructions for the package.

6. Run any of the following commands in cmd from the project root directory:

    * build - build the project into an executable
    * clean - clean build directory
    * run - run the code on the host pc
    * lint - run lint checking
    * generate-doc - generate documentation

7. To update the list of packages saved in requirements.txt, while venv is activated, from the root folder, run: pip list --format=freeze > requirements.txt

8. To deactivate the venv, run: deactivate

## Implementation Description

Refer to CCC-PPP-SW-XXX Implementation Description.

## Instructions to Use the Program

TODO
