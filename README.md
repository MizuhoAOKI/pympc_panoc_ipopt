# MPC implementation with Python, solved by PANOC/IPOPT
This repository provides samples of Model Predictive Controller(MPC) implementation with python. 

The options of optimization solver is PANOC and IPOPT. 

Just change "solver_type" in simulation_setting.yaml as you like. 

## What is IPOPT?
It is a famous solver for a nonlinear optimization problem.
See [IPOPT official page](https://coin-or.github.io/Ipopt/) for detail.

## What is OpEn?
It is a numerical optimization solver written in Rust. The algorythm is called [PANOC](https://arxiv.org/abs/1709.06487)(Proximal Averaged Newton-type method for Optimal Control).

Following features are highlighted.
- Embeddable
- Accurate
- Fast
- User Friendly
- Community
- Documented

See [Official Page](https://alphaville.github.io/optimization-engine/) for detail.


## Simulation Results
### Simple Pathtrack with MPC, predicting vehicle behavior by Kinematic Bicycle Model.
#### SVG Animation
![](./media/sample_result_pathtrack/trajectory.svg)
#### MPC log
![](./media/sample_result_pathtrack/mpc_result.png)
#### Simulator log
![](./media/sample_result_pathtrack/simulator_result.png)
#### Realtime visualizer
https://user-images.githubusercontent.com/63337525/168445716-ba20d188-c391-4b92-8a79-194076ac7a5b.mp4


<!--
## Environment costruction
Check latest information at [the official page](https://alphaville.github.io/optimization-engine/docs/installation).

### Install python, rust, and clang
- Install [Rust](https://www.rust-lang.org/tools/install)
- Install [Python](https://www.python.org/)
    - Use python 3.x
- Install [Clang](https://github.com/rust-lang/rust-bindgen/blob/master/book/src/requirements.md)

#### Especially for windows users who have [scoop](https://scoop.sh/), just run following commands.
- ```scoop install python```
- ```scoop install rustup```

#### Install latest opengen
- ```git clone https://github.com/alphaville/optimization-engine.git```
- ```cd optimization-engine/open-codegen```
- ```python setup.py install```
- ```cd ..``` (move to optimization-engine/)
- ```cargo build```

#### Install other python packages
- ```pip install -r requirements.txt```
-->

### Environment construction

Python version 3.8.x is recommended.

For ubuntu users, 

- Install [Python](https://www.python.org/)
    - [pyenv](https://github.com/pyenv/pyenv) is helpful to switch version of python interpreters.
- Install Poetry  
    - `$ curl -sSL https://install.python-poetry.org | python3 -`  
    - `$ echo export PATH="/home/mizuho/.local/bin:$PATH" >> ~/.bashrc`
- Install cargo  
    - `$ sudo apt-get -y install cargo`
- Clone this repository
    - `git clone https://github.com/MizuhoAOKI/pympc_panoc_ipopt.git`
- Install python libraries
    - ```$ cd pympc_panoc_ipopt```
    - ```$ poetry install ```

### How to run
```$ cd pympc_panoc_ipopt```  
```$ poetry shell```  
```$ cd pathtrack```  
```$ python main.py```  

## References
- [OpEn official HP](https://alphaville.github.io/optimization-engine/)
- [About installation of OpEn](https://alphaville.github.io/optimization-engine/docs/installation)
- [Python Examples of OpEn](https://alphaville.github.io/optimization-engine/docs/python-examples)
