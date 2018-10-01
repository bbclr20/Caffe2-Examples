import numpy as np
import os
import shutil
import operator
import glob
from caffe2.python import core,model_helper,optimizer,workspace,brew,utils
from caffe2.proto import caffe2_pb2
import matplotlib.pyplot as plt
from caffe2.python.modeling import initializers
from caffe2.python.modeling.parameter_info import ParameterTags

#
# Check files
#
# Train and test lmdb
TRAIN_LMDB = os.path.join("tutorial_data", "cifar10", "training_lmdb")
TEST_LMDB = os.path.join("tutorial_data", "cifar10", "testing_lmdb")

# Extract protobuf files from most recent Part 1 run
part1_runs_path = os.path.join("tutorial_files", "tutorial_cifar10")
runs = sorted(glob.glob(part1_runs_path + "/*"))

# Init and Predict net
INIT_NET = os.path.join(runs[-1], "cifar10_init_net.pb")
PREDICT_NET = os.path.join(runs[-1], "cifar10_predict_net.pb")

# Make sure they all exist
if (not os.path.exists(TRAIN_LMDB)) or (not os.path.exists(TEST_LMDB)) or (not os.path.exists(INIT_NET)) or (not os.path.exists(PREDICT_NET)):
    print("ERROR: input not found!")
else:
    print("Success, you may continue!")


def AddInputLayer(model, batch_size, db, db_type):
    data_uint8, label = model.TensorProtosDBInput(
        [],
        ["data_uint8", "label"],
        batch_size=batch_size,
        db=db,
        db_type=db_type
    )

    data = model.Cast(data_uint8, "data", to=core.DataType.FLOAT)
    # data = model.Scale(data, data, scale=float(1. / 256))  # scaled in download_data.py
    data = model.StopGradient(data, data)
    return data, label


def update_dims(height, width, kernel, stride, pad):
    new_height = ((height - kernel + 2 * pad) // stride) + 1
    new_width = ((width - kernel + 2 * pad) // stride) + 1
    return new_height, new_width


def Add_Original_CIFAR10_Model(model, data, num_classes, image_height, image_width, image_channels):
    conv1 = brew.conv(model, data, 'conv1', dim_in=image_channels, dim_out=32, kernel=5, stride=1, pad=2)
    h, w = update_dims(height=image_height, width=image_width, kernel=5, stride=1, pad=2)
    pool1 = brew.max_pool(model, conv1, 'pool1', kernel=3, stride=2)
    h, w = update_dims(height=h, width=w, kernel=3, stride=2, pad=0)
    relu1 = brew.relu(model, pool1, 'relu1')

    conv2 = brew.conv(model, relu1, 'conv2', dim_in=32, dim_out=32, kernel=5, stride=1, pad=2)
    h, w = update_dims(height=h, width=w, kernel=5, stride=1, pad=2)
    relu2 = brew.relu(model, conv2, 'relu2')
    pool2 = brew.average_pool(model, relu2, 'pool2', kernel=3, stride=2)
    h, w = update_dims(height=h, width=w, kernel=3, stride=2, pad=0)

    conv3 = brew.conv(model, pool2, 'conv3', dim_in=32, dim_out=64, kernel=5, stride=1, pad=2)
    h, w = update_dims(height=h, width=w, kernel=5, stride=1, pad=2)
    relu3 = brew.relu(model, conv3, 'relu3')
    pool3 = brew.average_pool(model, relu3, 'pool3', kernel=3, stride=2)
    h, w = update_dims(height=h, width=w, kernel=3, stride=2, pad=0)

    fc1 = brew.fc(model, pool3, 'fc1', dim_in=64 * h * w, dim_out=64)
    fc2 = brew.fc(model, fc1, 'fc2', dim_in=64, dim_out=num_classes)

    softmax = brew.softmax(model, fc2, 'softmax')
    return softmax


def AddTrainingOperators(model, softmax, label):
    xent = model.LabelCrossEntropy([softmax, label], 'xent')
    loss = model.AveragedLoss(xent, "loss")

    model.AddGradientOperators([loss])
    optimizer.build_sgd(
        model,
        base_learning_rate=0.1,
        policy="step",
        stepsize=10,
        gamma=0.999,
    )


def AddAccuracy(model, softmax, label):
    accuracy = brew.accuracy(model, [softmax, label], "accuracy")
    return accuracy


if __name__ == "__main__":
    arg_scope = {"order": "NCHW"}

    #
    # Restore test_model
    # Load init and predict nets
    #
    test_model = model_helper.ModelHelper(name="test_model", arg_scope=arg_scope, init_params=False)
    data, _ = AddInputLayer(test_model, 1, TEST_LMDB, "lmdb")

    init_net_proto = caffe2_pb2.NetDef()
    with open(INIT_NET, "rb") as f:
        init_net_proto.ParseFromString(f.read())
    test_model.param_init_net = test_model.param_init_net.AppendNet(core.Net(init_net_proto))

    predict_net_proto = caffe2_pb2.NetDef()
    with open(PREDICT_NET, "rb") as f:
        predict_net_proto.ParseFromString(f.read())
    test_model.net = test_model.net.AppendNet(core.Net(predict_net_proto))

    accuracy = brew.accuracy(test_model, ["softmax", "label"], "accuracy")

    #
    # Test loop
    #
    workspace.RunNetOnce(test_model.param_init_net)
    workspace.CreateNet(test_model.net, overwrite=True)

    avg_accuracy = 0.0
    test_iters = 10000

    for i in range(test_iters):
        workspace.RunNet(test_model.net)
        acc = workspace.FetchBlob('accuracy')
        avg_accuracy += acc
        if (i+1) % 500 == 0:
            print("Iter: {}, Current Accuracy: {}".format(i+1, avg_accuracy/float(i)))

    print("*********************************************")
    print("Final Test Accuracy: {}\n".format(avg_accuracy/float(test_iters)))

    #
    # Training and improving the model
    #
    training_iters = 3000
    # Reset workspace to clear all of the information from the testing stage
    workspace.ResetWorkspace()

    # Create new model
    arg_scope = {"order": "NCHW"}
    train_model = model_helper.ModelHelper(name="cifar10_train", arg_scope=arg_scope, init_params=False)

    data, label = AddInputLayer(train_model, 100, TRAIN_LMDB, "lmdb")
    softmax = Add_Original_CIFAR10_Model(train_model, data, 10, 32, 32, 3)
    # Populate the param_init_net of the model obj with the contents of the init net
    init_net_proto = caffe2_pb2.NetDef()
    with open(INIT_NET, "rb") as f:
        init_net_proto.ParseFromString(f.read())
    tmp_init_net = core.Net(init_net_proto)
    train_model.param_init_net = train_model.param_init_net.AppendNet(tmp_init_net)

    AddTrainingOperators(train_model, softmax, label)
    AddAccuracy(train_model, softmax, label)

    #
    # Training loop
    #
    # Prime the workspace
    workspace.RunNetOnce(train_model.param_init_net)
    workspace.CreateNet(train_model.net, overwrite=True)

    # Run the training loop
    for i in range(training_iters):
        workspace.RunNet(train_model.net)
        acc = workspace.FetchBlob('accuracy')
        loss = workspace.FetchBlob('loss')
        if (i+1) % 500 == 0:
            print("Iter: {}, Loss: {}, Accuracy: {}".format(i+1, loss, acc))

    #
    # Get confusion matrix
    #
    arg_scope = {"order": "NCHW"}
    test_model = model_helper.ModelHelper(name="test_model", arg_scope=arg_scope, init_params=False)
    data, label = AddInputLayer(test_model, 1, TEST_LMDB, "lmdb")
    softmax = Add_Original_CIFAR10_Model(test_model, data, 10, 32, 32, 3)
    AddAccuracy(test_model, softmax, label)
    # accuracy = brew.accuracy(test_model, ['softmax', 'label'], 'accuracy')

    workspace.RunNetOnce(test_model.param_init_net)
    workspace.CreateNet(test_model.net, overwrite=True)

    # Confusion Matrix for CIFAR-10
    cmat = np.zeros((10, 10))
    avg_accuracy = 0.0
    test_iters = 10000
    for i in range(test_iters):
        workspace.RunNet(test_model.net)
        acc = workspace.FetchBlob("accuracy")
        avg_accuracy += acc
        if (i % 500 == 0) and (i > 0):
            print("Iter: {}, Current Accuracy: {}".format(i, avg_accuracy / float(i)))

        # Get the top-1 prediction
        results = workspace.FetchBlob("softmax")[0]
        label = workspace.FetchBlob("label")[0]
        max_index, max_value = max(enumerate(results), key=operator.itemgetter(1))
        # Update confusion matrix
        cmat[label, max_index] += 1

    print("*********************************************")
    print("Final Test Accuracy: ", avg_accuracy / float(test_iters))

    #
    # Plot confusion matrix
    #
    fig = plt.figure(figsize=(10,10))
    plt.tight_layout()
    ax = fig.add_subplot(111)
    res = ax.imshow(cmat, cmap=plt.cm.rainbow, interpolation='nearest')
    width, height = cmat.shape
    for x in range(width):
        for y in range(height):
            ax.annotate(str(cmat[x, y]), xy=(y, x), horizontalalignment='center', verticalalignment='center')

    classes = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
    plt.xticks(range(width), classes, rotation=0)
    plt.yticks(range(height), classes, rotation=0)
    ax.set_xlabel('Predicted Class')
    ax.set_ylabel('True Class')
    plt.title('CIFAR-10 Confusion Matrix')
    plt.show()
