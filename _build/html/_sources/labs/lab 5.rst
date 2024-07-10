Lab5: Get It Together
=====================

Goals of this lab
-----------------

- `Implement-80% <#implement>`__ 
- `Results-10% <#results>`__
- `Analyze-10% <#analyze>`__

Introduction
------------

After we finished the Lab3, the implementation of Systolic Array, we
want to use it to run an actual Machine Learning Model to let you
understand the true bottleneck when you are designing a Chip to
accelerate your workloads. And the Lab3 focuses on the compute unit, and
in this lab, you will need to understand how software can use the chip
you designed and accelerate the matrix multiply or even any operator
that is a heavy burden on the CPU.

From the class, you’ve learned the Von Neumann Architecture which is
inefficient on compute-intensive workloads, such as matrix multiply,
Convolution, Weather simulation, and also the most popular application,
Computer Vision. And this is just a theory that you’ve learned that you
need to increase the opportunity of data reuse and decrease the data
movement. This is the best chance to let you truly understand how to
improve the workloads.

Before diving into the Lab, you need to understand the following
concept.

1. How does a CPU offload the tasks to the Accelerator?
2. How to fully utilize PEs on the accelerator.

Intended Learning Outcomes
--------------------------

-  Understanding the architecture of cpu and cfu,and how they
   communicate with each other.
-  Adding the systolic array architecture within the cfu-playgrond.

Background
----------

The matrix data needs to be transmitted from the CPU to the global
buffers A and B in the CFU. Once all the required data has been
gathered, then the TPU will start to compute this matrix data. The
outcome of the computation will be preserved in the buffer C. Finally,
the data stored in the buffer C will be written back to the CPU.

.. image:: https://hackmd.io/_uploads/Sk2gIodza.jpg

cfu-cpu communication
^^^^^^^^^^^^^^^^^^^^^

The “CFU bus” provides the communication between the CPU and CFU. The
CFU Bus is composed of two independent streams:

-  The CPU uses the command stream (cmd) to send operands and 10 bits of
   function code to the CFU, thus initiating the CFU computation.

-  The CFU uses the response stream (rsp) to return the result to the
   CPU. Since the responses are not tagged, they must be delivered
   in-order.

Each stream has two-way handshaking and backpressure (\*_valid and
\*_ready in the diagram below). An endpoint can indicate that it cannot
accept another transfer by pulling its ready signal low. A transfer
takes place only when both valid from the sender and ready from the
receiver are high.

.. hint::

   he data values from the CPU (cmd_function_id, cmd_inputs_0,
   and cmd_inputs_1) are valid ONLY during the cycle that the handshake
   is active (when both cmd_valid and cmd_ready are asserted). If your
   CFU needs to use these values in subsequent cycles, it must store
   them in registers.

.. image::
   ./images/rJ4Or83A3.png

.. image::
   ./images/r1T4UwpRh.png

`soucefrom <https://cfu-playground.readthedocs.io/en/latest/step-by-step.html>`__

how to define cfu_op
^^^^^^^^^^^^^^^^^^^^

.. image::
   ./images/BkjuuUnC2.png

``CFU-Playground/common/src/cfu.h`` Here defines cfu_op, 
the interface between cfu and cpu by

1. cmd_function_id[ 9:0 ]
2. cmd_inputs_0[ 31:0 ] 
3. cmd_inputs_1[ 31:0 ] 
4. rsp_outputs_0[ 31:0 ]

- cmd_function_id[ 2:0 ] and cmd_function_id[ 9:3 ] corresponding to funct3,funct7 respectively.
- cmd_inputs_0,1 corresponding to rs1,rs2 respectively.
- rsp_outputs_0 is the return value of cfu_op

where to devise
^^^^^^^^^^^^^^^

``CFU-Playground/proj/"your project"/src/tensorflow/lite/kernels/internal/reference/integer_ops/conv.h``

``CFU-Playground/proj/"your project name"/cfu.v``

Your Assignment
^^^^^^^^^^^^^^^

Implement
"""""""""

-  Objective :

   -  Implementing a systolic array to accelerate convolution
      operations.

-  Input :

   -  2-D Matrix from Im2Col

-  Output :

   -  The result computed from the systolic array in the CFU

Results
"""""""

-  Show the results of the labels of KWS model after quantized ,and see
   if they match those of lab2. 
   
.. image::
   ./images/H1QUcXCWa.png

Analyze
"""""""

-  please compare the total cycles to the results of lab1 and lab2 and
   analyze it

debug
"""""

``CFU-Playground/proj/"your project name"/src/proj_menu.cc`` You can
add a new function here, and then run “make PLATFORM=sim load,” select
“t->3->function you defined,” and this will generate a waveform (.vcd)
file stored in soc/build/sim.xxx/gateware/sim.vcd .

