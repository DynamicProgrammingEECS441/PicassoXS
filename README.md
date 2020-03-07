# ***PicassoXS***
EECS 441 @ UMich Project 

## *Description*
PicassoXS is your professional photo editing app that transfer your own photo into different painting

## *Repo Structure*

```shell
.
├── README.md
├── doc    # DOCUMENTATION
│   ├── AlgorithmSelection    # doc for algorithm selection 
│   │   └── README.md
│   └── PaintingSelection    # doc for painting selection 
│       └── README.md
└── src    # CODE 
    ├── backend    # code for BACKEND 
    │   ├── layers.py
    │   ├── main.py
    │   ├── module.py
    │   ├── tfserver.    # code for using TF Server 
    │   │   ├── Instruction.md
    │   │   ├── servable    # servile model 
    │   │   │   ├── model1
    │   │   │   └── model2
    │   │   └── servable_demo.ipynb	
    │   └── train.py
    └── frontend    # code for FRONTEND		
        └── PicassoXS
```

## *Contributors*
### Frontend
- Zhijie Ren (uniqname: rzj)
- Senhao Wang (uniqname: rogerogw)
- Juye Xiao (uniqname: juyexiao)

### Backend
- Xiao Song (uniqname: xiaosx)
- Enhao Zhang (uniqname: ehzhang)
- Rena Xu (uniqname: renaxu)
- Jason Zhao (uniqname: jasonyz)


## *Development Environment*
### ***IDE***: 
- Frontend - Xcode

- Backend - Visual Studio Code

### ***Programming Languages:***
- Frontend - Swift
- Backend - Python v.3.5 ~ v.3.7

### ***Operating System:***
- Ubuntu 18.04 LTS


## *Third Party Libraries*
## 1. TensorFlow
Release: 2.0

TensorFlow is an end-to-end open source platform for machine learning. We can use it for all model traning.

Installation

`$ pip install tensorflow`

[TensorFlow intallation](https://www.tensorflow.org/install)

[TensorFlow documentation](https://www.tensorflow.org/api_docs)

## 2. Docker
Release: 19.03.5

Docker provides a way to run applications securely isolated in a container, packaged with all its dependencies and libraries. We can use Docker's *containers* to create virtual environments that isolate a TensorFlow installation from the rest of the system.

***Note:*** Docker Desktop version is better, but can still install through command line

Installation
```diff
$ curl -fsSL https://get.docker.com -o get-docker.sh
$ sudo sh get-docker.sh
```
note: needs to log out and log back in for docker to work

[Docker installation](https://docs.docker.com/install/)

[Docker documentation](https://docs.docker.com/)

##  3. gRPC
Release: 1.27.1

Google's remote process call framwork. It can efficiently connect services in and across data centers with pluggable support for load balancing, tracing, health checking and authentication.

Installation

`$ pip install grpcio`

[gRPC installation](https://grpc.io/blog/installation/)

[gRPC documentation](https://grpc.io/docs/)

### Protocal Buffers

- .proto extension

- A language-neutral, platform-neutral, extensible way of serializing structured data for use in communications protocols, data storage, and more

- Used with gRPC

- [Protocal Buffer documentation](https://developers.google.com/protocol-buffers/docs/overview)

## 4. AWS
EC2 (Elastic Compute Cloud): 
- Free to use
- Designed to make web-scale cloud computing easier for developers. Amazon EC2’s simple web service interface allows you to obtain and configure capacity with minimal friction
- [EC2 documentation](https://docs.aws.amazon.com/ec2/index.html?nc2=h_ql_doc_ec2) 

ECS (Elastic Container Service):
- A fully managed container orchestration service
- Manage Docker containers on a cluster
- [ECS documentation](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/Welcome.html)


## 5. OpenCV
Release: 4.2.0

A library of programming functions mainly aimed at real-time computer vision.

Installation

`$ pip install opencv-python`

[OpenCV installation](https://linuxize.com/post/how-to-install-opencv-on-ubuntu-18-04/)

[OpenCV documentation](https://docs.opencv.org/master/d9/df8/tutorial_root.html)

## 6. SQLAlchemy
Release: 1.3.13

SQLAlchemy is the Python SQL toolkit and Object Relational Mapper that gives application developers the full power and flexibility of SQL.

Installation

`$ pip install SQLAlchemy`

[SQLAlchemy installation](https://pypi.org/project/SQLAlchemy/)

[SQLAlchemy documentation](https://docs.sqlalchemy.org/en/13/)


## 7. PyTorch
Release: 1.4.0

PyTorch is an open source machine learning library based on the Torch library, used for applications such as computer vision and natural language processing.

Installation

`$ pip install torch torchvision`

[PyTorch installation](https://pytorch.org/get-started/locally/)

[PyTorch documentation](https://pytorch.org/docs/stable/index.html)

## 8. Tornado
Release: 6.0.3

Tornado is a Python web framework and asynchronous networking library

Installation

`$ pip install tornado`

[Tornado installation](https://www.tornadoweb.org/en/stable/)

[Tornado documentation](https://www.tornadoweb.org/en/stable/guide/intro.html)

## 9. tqdm
Release: 4.43.0

A progress bar library with good support for nested loops

Installation

`$ pip install tqdm`

[tqdm documentation](https://tqdm.github.io/)

## 10. Pillow
Release: 2.2.2

Python Image Library. A free library for the Python programming language that adds support for opening, manipulating, and saving many different image file formats

Installation

`$ pip install Pillow==2.2.2`

[Pillow documentation](https://pillow.readthedocs.io/en/stable/)

## 11. SpiPy
Release: 1.4.1

A python-based ecosystem of open-source software for mathematics, science, and engineering

Installation

`$ pip install scipy`

[SciPy documentation](https://www.scipy.org/docs.html)

## 12. imageio
Release: 2.8.0

Provides an easy interface to read and write a wide range of image data

Installation

`$ pip install imageio`

[imageio documentation](https://imageio.readthedocs.io/en/stable/)

## 13. Tensorflow Addons
Release: 0.8.2

TensorFlow Addons is a repository of contributions that conform to well- established API patterns, but implement new functionality not available in core TensorFlow

Installation

`$ pip install tensorflow-addons`





## *Set Up Third Party Dependency*
- **dep_install executable**: one script that install dependencies completely
- **pip_setup executable**: install python 3.7 and pip 20.0.2
- **requirements.txt**: list thrid party libaries for pip to install

***Note:***
- Your environment should be ubuntu 18.04 before you run the following command
- You might have to press *yes* or *Enter* several times during installation to give permission to install
- If you wish to install the libraries manually, be sure to run upgrade pip3

    -  `$ pip3 install --upgrade pip`


***Library setup:***

**Method 1:** 
Install with pip and requirements.txt

1. Upgrade python to python3 and upgrade pip to pip 20.0.2
    - `$ chmod +x pip_setup`
    - `$ ./pip_setup`

2. Install libraries with pip and requirements.txt 
    - `$ pip install -r requirements.txt`


**Method 2:**
Install with shell script
1. Give the following shell script permission
    - `$ chmod +x dep_install`
2. Run the script
    - `$ ./dep_install`

**Method 3:**
Install manually
1. Upgrade python to python3 and upgrade pip to pip 20.0.2
    - `$ chmod +x pip_setup`
    - `$ ./pip_setup`
2. Run each installment command in the **Third Party Libraries section**

