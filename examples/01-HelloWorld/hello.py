#!/usr/bin/python3
from caffe2.python import (
    workspace,
    model_helper,
    net_drawer,
)
import numpy as np

# Create the input data, 16 is the size of the batch
data = np.random.rand(16, 100).astype(np.float32)

# Create labels for the data as integers [0, 9].
label = (np.random.rand(16) * 10).astype(np.int32)

workspace.FeedBlob("data", data)
workspace.FeedBlob("label", label)

# Create model using a model helper
m = model_helper.ModelHelper(name="my first net")
weight = m.param_init_net.XavierFill([], 'fc_w', shape=[10, 100])
bias = m.param_init_net.ConstantFill([], 'fc_b', shape=[10, ])

fc_1 = m.net.FC(["data", "fc_w", "fc_b"], "fc1")
pred = m.net.Sigmoid(fc_1, "pred")
softmax, loss = m.net.SoftmaxWithLoss([pred, "label"], ["softmax", "loss"])

# print(m.net.Proto())
# print(m.param_init_net.Proto())

# save the model as graph
graph = net_drawer.GetPydotGraph(m.net, rankdir="BT")
graph.write_png("hello.png")

# init, create and run
m.AddGradientOperators([loss])  # add gradient
# print(m.net.Proto())          # observe gradient

workspace.RunNetOnce(m.param_init_net)
workspace.CreateNet(m.net)

for ii in range(100):
    data = np.random.rand(16, 100).astype(np.float32)
    label = (np.random.rand(16) * 10).astype(np.int32)

    workspace.FeedBlob("data", data)
    workspace.FeedBlob("label", label)

    workspace.RunNet(m.name, 10)   # run for 10 times
    # print("Run: ", ii)

# save the model with grad
graph = net_drawer.GetPydotGraph(m.net, rankdir="BT")
graph.write_png("hello_with_grad.png")
