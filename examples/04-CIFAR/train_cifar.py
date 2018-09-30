from matplotlib import pyplot as plt
import numpy as np
import os
from caffe2.python.predictor import mobile_exporter
from caffe2.python import (
    brew,
    core,
    model_helper,
    net_drawer,
    optimizer,
    workspace,
)
import datetime
import math


def AddInput(model, batch_size, db, db_type):
    # load the data
    data_uint8, label = brew.db_input(
        model,
        blobs_out=["data_uint8", "label"],
        batch_size=batch_size,
        db=db,
        db_type=db_type,
    )
    data = model.Cast(data_uint8, "data", to=core.DataType.FLOAT)
    # data = model.Scale(data, data, scale=float(1./256))  # scaled in download_data.py
    data = model.StopGradient(data, data)
    return data, label


def update_dims(height, width, kernel, stride, pad):
    new_height = ((height - kernel + 2 * pad) // stride) + 1
    new_width = ((width - kernel + 2 * pad) // stride) + 1
    return new_height, new_width


def Add_Original_CIFAR10_Model(model, data, num_classes, image_height, image_width, image_channels, save_png=True):
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
    if save_png:
        graph = net_drawer.GetPydotGraph(model.net, rankdir="LR")
        graph.write_png("CIFAR10_Model.png")
    return softmax


def AddTrainingOperators(model, softmax, label, save_png=True):
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

    if save_png:
        graph = net_drawer.GetPydotGraph(model.net, rankdir="LR")
        graph.write_png("CIFAR10_with_Grad.png")


def AddAccuracy(model, softmax, label):
    accuracy = brew.accuracy(model, [softmax, label], "accuracy")
    return accuracy


# Add checkpoints to a given model
def AddCheckpoints(model, checkpoint_iters, db_type):
    ITER = brew.iter(model, "iter")
    db = os.path.join(unique_timestamp, "cifar10_checkpoint_%05d.lmdb")
    model.Checkpoint([ITER] + model.params, [], db=db,
                     db_type=db_type, every=checkpoint_iters)


if __name__ == "__main__":
    # Paths to the init & predict net output locations
    init_net_out = 'cifar10_init_net.pb'
    predict_net_out = 'cifar10_predict_net.pb'

    # Dataset specific params
    image_width = 32
    image_height = 32
    image_channels = 3
    num_classes = 10

    # Training params
    training_iters = 3000
    training_net_batch_size = 100
    validation_images = 6000  # set in download_data.py
    validation_interval = 100
    checkpoint_iters = 1000

    root_folder = os.path.join('tutorial_files', 'tutorial_cifar10')
    if not os.path.isdir(root_folder):
        os.makedirs('tutorial_files')
        os.makedirs(root_folder)
    workspace.ResetWorkspace(root_folder)

    # Create uniquely named directory under root_folder to output checkpoints to
    unique_timestamp = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    checkpoint_dir = os.path.join(root_folder, unique_timestamp)
    os.makedirs(checkpoint_dir)
    print("Checkpoint output location: ", checkpoint_dir)

    #
    # TRAINING MODEL
    # Initialize with ModelHelper class
    #
    arg_scope = {"order": "NCHW"}
    train_model = model_helper.ModelHelper(
        name="train_net", arg_scope=arg_scope)

    training_lmdb_path = os.path.join("tutorial_data", "cifar10", "training_lmdb")
    data, label = AddInput(
        train_model, batch_size=training_net_batch_size,
        db=training_lmdb_path,
        db_type='lmdb')

    softmax = Add_Original_CIFAR10_Model(train_model, data, num_classes, image_height, image_width, image_channels)
    AddTrainingOperators(train_model, softmax, label)
    AddCheckpoints(train_model, checkpoint_iters, db_type="lmdb")

    #
    # VALIDATION MODEL
    # Initialize with ModelHelper class without re-initializing params
    #
    val_model = model_helper.ModelHelper(
        name="val_net", arg_scope=arg_scope, init_params=False)

    validation_lmdb_path = os.path.join("tutorial_data", "cifar10", "validation_lmdb")
    data, label = AddInput(
        val_model, batch_size=validation_images,
        db=validation_lmdb_path,
        db_type='lmdb')

    softmax = Add_Original_CIFAR10_Model(val_model, data, num_classes, image_height, image_width, image_channels)
    AddAccuracy(val_model, softmax, label)

    #
    # DEPLOY MODEL
    # Initialize with ModelHelper class without re-initializing params
    #
    deploy_model = model_helper.ModelHelper(
        name="deploy_net", arg_scope=arg_scope, init_params=False)
    Add_Original_CIFAR10_Model(deploy_model, "data", num_classes, image_height, image_width, image_channels)
    print("Training, Validation, and Deploy models all defined!")

    #
    # start training
    #
    workspace.RunNetOnce(train_model.param_init_net)
    workspace.CreateNet(train_model.net, overwrite=True)
    workspace.RunNetOnce(val_model.param_init_net)
    workspace.CreateNet(val_model.net, overwrite=True)

    # track loss and validation accuracy
    loss = np.zeros(int(math.ceil(training_iters / validation_interval)))
    val_accuracy = np.zeros(int(math.ceil(training_iters / validation_interval)))
    val_count = 0
    iteration_list = np.zeros(int(math.ceil(training_iters / validation_interval)))

    for i in range(training_iters):
        workspace.RunNet(train_model.net)

        if i % validation_interval == 0:
            print("Training iter: ", i)
            loss[val_count] = workspace.FetchBlob('loss')
            workspace.RunNet(val_model.net)
            val_accuracy[val_count] = workspace.FetchBlob('accuracy')
            print("Loss: ", str(loss[val_count]))
            print("Validation accuracy: ", str(val_accuracy[val_count]) + "\n")
            iteration_list[val_count] = i
            val_count += 1

    plt.figure("Loss and Accuracy")
    plt.title("Training Loss vs. Validation Accuracy")
    plt.plot(iteration_list, loss, 'b')
    plt.plot(iteration_list, val_accuracy, 'r')
    plt.xlabel("Training iteration")
    plt.legend(('Loss', 'Validation Accuracy'), loc='upper right')
    plt.show()

    #
    # save model
    #
    workspace.RunNetOnce(deploy_model.param_init_net)
    workspace.CreateNet(deploy_model.net, overwrite=True)

    # Use mobile_exporter's Export function to acquire init_net and predict_net
    init_net, predict_net = mobile_exporter.Export(workspace, deploy_model.net, deploy_model.params)

    full_init_net_out = os.path.join(checkpoint_dir, init_net_out)
    full_predict_net_out = os.path.join(checkpoint_dir, predict_net_out)

    # Simply write the two nets to file
    with open(full_init_net_out, 'wb') as f:
        f.write(init_net.SerializeToString())
    with open(full_predict_net_out, 'wb') as f:
        f.write(predict_net.SerializeToString())
    print("Model saved as " + full_init_net_out + " and " + full_predict_net_out)
