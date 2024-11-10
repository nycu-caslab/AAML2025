# Final Project: MLPerf™ Tiny

TBA

<!-- ## Problem
In the final project, you are required to design a CFU for MLPerf™ Tiny image classification benchmark model and targeting on decreasing latency.  

Also, Your design will be benchmarked by the MLPerf™ Tiny Benchmark Framework. Here is its [Github page](https://github.com/mlcommons/tiny) for detailed information aboud MLPerf™ Tiny.

### Selected Model

We use [MLPerf™ Tiny Image Classification Benchmark Model](https://github.com/mlcommons/tiny/tree/master/benchmark/training/image_classification) for the project. It is a tiny version of ResNet, consisting of Conv2D, Add, AvgPool2D, FC, and Softmax.

You don't need to itegrate the model on yourself. The model is already included in CFU. See `${CFU_ROOT}/common/src/models/mlcommons_tiny_v01/imgc/`. Also, you can inspect the architecture of the selected model with [Netron](https://netron.app/). It might provide you some inspiration for your design.

## Setup


## Requirements

### Files
You can modify the following files:
* Kernel API
    1. `tensorflow/lite/micro/kernels/add.cc`
    2. `tensorflow/lite/micro/kernels/conv.cc`
    3. `tensorflow/lite/micro/kernels/fully_connected.cc`
    4. `tensorflow/lite/micro/kernels/softmax.cc`
    5. `tensorflow/lite/micro/kernels/pooling.cc`

* Kernel Implementation
    1. `tensorflow/lite/kernels/internal/reference/integer_ops/add.h`
    2. `tensorflow/lite/kernels/internal/reference/integer_ops/conv.h`
    3. `tensorflow/lite/kernels/internal/reference/integer_ops/fully_connected.h`
    4. `tensorflow/lite/kernels/internal/reference/integer_ops/softmax.h`
    5. `tensorflow/lite/kernels/internal/reference/integer_ops/pooling.h`

* HW design
    1. `cfu.v`

```{important}
No other source code in `${CFU_ROOT}/common/**` and `${CFU_ROOT}/third_party/**` should be overriden unless asking for permission.
```

### Golden Test
Secondly, your design should pass the golden test. 
* After `make prog && make load`, input `11g` to run golden test of MLPerf Tiny imgc model. The result should be like:  
![](images/golden.png)

### Architecture
You can also modify the architecture or the parameters of the selected model. The classification accuracy of your design is evaluated.

However, **DO NOT RETRAIN THE MODEL ON TESTING IMAGES**.

### Performance
We use the evaluation script to evaluate your design.

* Usage:
    * `make prog && make load` > reboot litex > turn off litex-term > run eval script
    * `python eval_script.py` in `${CFU_ROOT}/proj/AAML_final_pro`
        * `--port {tty_path:-/dev/ttyUSB1}`: Add this argument to select correct serial port

The result should be like:  
![](images/script.png)

Improve the performance of your design to decrease the latency as low as it could be.

```{note}
If you just want to know the latency of your design, it would be easier to run a test input instead of whole process of evaluation.
```

## Presentation - 30%
```{important}
You will receive 0 point if you don't present your work
```
The presentation takes 30% of your final project score. -->