#!/usr/bin/python3
import urllib
import pandas
import numpy as np
from matplotlib import pyplot as plt
from io import BytesIO, StringIO
import os
from caffe2.python import (
    core,
    utils,
    workspace,
    model_helper,
    net_drawer,
    brew,
    optimizer
)
from caffe2.proto import caffe2_pb2

#
# Download and display data
#
f = urllib.request.urlopen(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data")
raw_data = f.read()
print("\n===== Raw data looks like this =====\n")
print(raw_data[:100].decode("ascii") + '...')
df = pandas.read_csv(BytesIO(raw_data), encoding="utf8", header=None)
print("\n===== Data frame by pandas =====\n{}\n".format(df.head(5)))

#
# Transform byte string to numerical data and labels
#
features = np.loadtxt(StringIO(raw_data.decode("utf-8")),
                      dtype=np.float32, delimiter=",", usecols=(0, 1, 2, 3))
for i in range(4):
    features[:, i] = (features[:, i] - features[:, i].min()) / (features[:, i].max() - features[:, i].min())
label_converter = lambda s: {b'Iris-setosa': 0, b'Iris-versicolor': 1, b'Iris-virginica': 2}[s]
labels = np.loadtxt(StringIO(raw_data.decode("utf-8")),
                    dtype=np.int, delimiter=',', usecols=(4,), converters={4: label_converter})

data_size = len(labels)
random_index = np.random.permutation(data_size)  # shuffle
features = features[random_index]
labels = labels[random_index]

train_features = features[:130]
train_labels = labels[:130]
test_features = features[130:]
test_labels = labels[130:]

#
# Visualize the data
#
plt.figure("Datasets")
plt.subplot(2, 1, 1)
legend = ["rx", "b+", "go"]
plt.title("Training data distribution, feature 0 and 1")
for i in range(3):
    plt.plot(train_features[train_labels == i, 0],
             train_features[train_labels == i, 1],
             legend[i])
plt.axis("image")
plt.xlim(-0.2, 1.2)
plt.ylim(-0.2, 1.2)

plt.subplot(2, 1, 2)
plt.title("Test data distribution, feature 0 and 1")
for i in range(3):
    plt.plot(test_features[test_labels == i, 0],
             test_features[test_labels == i, 1],
             legend[i])
plt.axis("image")
plt.xlim(-0.2, 1.2)
plt.ylim(-0.2, 1.2)
# plt.show()

# #
# # Create Tensor Protos
# #
# feature_and_label = caffe2_pb2.TensorProtos()
# feature_and_label.protos.extend([
#     utils.NumpyArrayToCaffe2Tensor(features[0]),
#     utils.NumpyArrayToCaffe2Tensor(labels[0])
# ])
# print("This is what tensor proto looks like:\n{}".format(feature_and_label))
# print("This is the compact string written into db:\n{}".
#       format(feature_and_label.SerializeToString()))


def write_db(db_type, db_name, features, labels):
    db = core.C.create_db(db_type, db_name, core.C.Mode.write)
    transaction = db.new_transaction()

    for i in range(features.shape[0]):
        feature_and_label = caffe2_pb2.TensorProtos()
        feature_and_label.protos.extend([
            utils.NumpyArrayToCaffe2Tensor(features[i]),
            utils.NumpyArrayToCaffe2Tensor(labels[i])
        ])

        transaction.put(
            "train_%03d".format(i),
            feature_and_label.SerializeToString()
        )
    del transaction
    del db


def AddInput(model, db, db_type, date_name, label_name):
    data, label = brew.db_input(
        model,
        blobs_out=[date_name, label_name],
        batch_size=100,
        db=db,
        db_type=db_type,
    )
    return data, label


def AddMLPModel(model, data):
    layer_sizes = [4, 10, 3]
    layer = data
    for i in range(len(layer_sizes) - 1):
        layer = brew.fc(model, layer, 'dense_{}'.format(i), dim_in=layer_sizes[i], dim_out=layer_sizes[i + 1])
        layer = brew.relu(model, layer, 'relu_{}'.format(i))
    softmax = brew.softmax(model, layer, 'softmax')
    return softmax


def AddAccuracy(model, softmax, label):
    """Adds an accuracy op to the model"""
    accuracy = brew.accuracy(model, [softmax, label], "accuracy")
    return accuracy


def AddTrainingOperators(model, softmax, label):
    """Adds training operators to the model."""
    xent = model.LabelCrossEntropy([softmax, label], 'xent')
    loss = model.AveragedLoss(xent, "loss")

    AddAccuracy(model, softmax, label)

    model.AddGradientOperators([loss])
    optimizer.build_adagrad(
        model,
        base_learning_rate=1e-1,
        policy="step",
        stepsize=1,
        gamma=0.9999,
    )


def VizResult(loss, accuracy):
    plt.figure("Summary of Training")
    plt.title("Summary of Training Run")
    plt.plot(loss, 'b')
    plt.plot(accuracy, 'r')
    plt.xlabel("Iteration")
    plt.legend(('Loss', 'Accuracy'), loc='upper right')
    plt.show()


if __name__ == "__main__":
    if not os.path.exists("data"):
        os.mkdir("data")
    write_db("minidb", "data/iris_train.minidb", train_features, train_labels)
    write_db("minidb", "data/iris_test.minidb", test_features, test_labels)

    #
    # Define a training model
    #
    train_model = model_helper.ModelHelper("iris_train")
    train_data, train_labels= AddInput(train_model, "data/iris_train.minidb", "minidb","train_data", "train_label")
    softmax = AddMLPModel(train_model, train_data)
    AddTrainingOperators(train_model, softmax, train_labels)

    graph = net_drawer.GetPydotGraph(train_model, rankdir="LR")
    graph.write_png("Iris_MLP.png")

    #
    # Run training
    #
    workspace.RunNetOnce(train_model.param_init_net)
    workspace.CreateNet(train_model.net, overwrite=True)
    total_iters = 150

    accuracy = np.zeros(total_iters)
    loss = np.zeros(total_iters)

    for i in range(total_iters):
        workspace.RunNet(train_model.net)
        accuracy[i] = workspace.blobs['accuracy']
        loss[i] = workspace.blobs['loss']
        # Check the accuracy and loss every so often
        if i % 10 == 0:
            print("Iter: {}, Loss: {}, Accuracy: {}".format(i, loss[i], accuracy[i]))

    VizResult(loss, accuracy)
