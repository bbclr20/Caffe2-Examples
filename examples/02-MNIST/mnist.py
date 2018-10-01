#!/usr/bin/python3
from caffe2.python import (
    brew,
    core,
    model_helper,
    net_drawer,
    optimizer,
    visualize,
    workspace,
)
from caffe2.python.predictor import predictor_exporter as pe
from caffe2.python.predictor import mobile_exporter as me

from matplotlib import pyplot as plt
import numpy as np
import operator
import os
import shutil


# If you would like to see some really detailed initializations,
# you can change --caffe2_log_level=0 to --caffe2_log_level=-1
core.GlobalInit(['caffe2', '--caffe2_log_level=0'])
print("Necessities imported!")


def DownloadResource(url, path):
    """Downloads resources from s3 by url and unzips them to the provided path"""
    import requests, zipfile, io
    print("Downloading... {} to {}".format(url, path))
    r = requests.get(url, stream=True)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(path)
    print("Completed download and extraction.")


#
# Prepare data and make folders
#
data_folder = os.path.join('tutorial_data', 'mnist')
root_folder = os.path.join('tutorial_files', 'tutorial_mnist')
print("training data folder:" + data_folder)
print("workspace root folder:" + root_folder)

db_missing = False

if not os.path.exists(data_folder):
    os.makedirs(data_folder)
    print("Your data folder was not found!! This was generated: {}".format(data_folder))

train_data = os.path.join(data_folder, "mnist-train-nchw-lmdb", "data.mdb")
train_lock = os.path.join(data_folder, "mnist-train-nchw-lmdb", "lock.mdb")
if os.path.exists(train_data) and os.path.exists(train_lock):
    print("lmdb train db found!")
else:
    db_missing = True

test_data = os.path.join(data_folder, "mnist-test-nchw-lmdb", "data.mdb")
test_lock =os.path.join(data_folder, "mnist-test-nchw-lmdb", "lock.mdb")
if os.path.exists(test_data) and os.path.exists(test_lock):
    print("lmdb test db found!")
else:
    db_missing = True

if db_missing:
    print("one or both of the MNIST lmbd dbs not found!!")
    db_url = "http://download.caffe2.ai/databases/mnist-lmdb.zip"
    try:
        DownloadResource(db_url, data_folder)
    except Exception as ex:
        print("Failed to download dataset. Please download it manually from {}".format(db_url))
        print("Unzip it and place the two database folders here: {}".format(data_folder))
        raise ex

# Clean up statistics from any old runs
if os.path.exists(root_folder):
    print("Looks like you ran this before, so we need to cleanup those old files...")
    shutil.rmtree(root_folder)

os.makedirs(root_folder)
workspace.ResetWorkspace(root_folder)


def AddInput(model, batch_size, db, db_type):
    """Load data with brew or model.TensorProtosDBInput"""
    # load the data from db - Method 1 using brew
    data_uint8, label = brew.db_input(
       model,
       blobs_out=["data_uint8", "label"],
       batch_size=batch_size,
       db=db,
       db_type=db_type,
    )

    # # load the data from db - Method 2 using TensorProtosDB
    # data_uint8, label = model.TensorProtosDBInput(
    #     [],
    #     ["data_uint8", "label"],
    #     batch_size=batch_size,
    #     db=db,
    #     db_type=db_type
    # )

    data = model.Cast(data_uint8, "data", to=core.DataType.FLOAT)
    data = model.Scale(data, data, scale=float(1./256))

    # don't need the gradient for the backward pass
    data = model.StopGradient(data, data)
    return data, label


def AddMLPModel(model, data, save_png=True):
    """Multi-layer Perceptron model"""
    size = 28 * 28 * 1
    sizes = [size, size * 4, size * 2, 10]
    layer = data
    for i in range(len(sizes) - 1):
        layer = brew.fc(model, layer, 'dense_{}'.format(i), dim_in=sizes[i], dim_out=sizes[i + 1])
        layer = brew.relu(model, layer, 'relu_{}'.format(i))
    softmax = brew.softmax(model, layer, 'softmax')

    if save_png:
        graph = net_drawer.GetPydotGraph(model.net)
        graph.write_png("MLP.png")

    return softmax


def AddLeNetModel(model, data, save_png=True):
    """This part is the standard LeNet model: from data to the softmax prediction.
    For each convolutional layer we specify dim_in - number of input channels
    and dim_out - number or output channels. Also each Conv and MaxPool layer changes the
    image size. For example, kernel of size 5 reduces each side of an image by 4.

    While when we have kernel and stride sizes equal 2 in a MaxPool layer, it divides
    each side in half.
    """
    conv1 = brew.conv(model, data, 'conv1', dim_in=1, dim_out=20, kernel=5)
    pool1 = brew.max_pool(model, conv1, 'pool1', kernel=2, stride=2)

    conv2 = brew.conv(model, pool1, 'conv2', dim_in=20, dim_out=50, kernel=5)
    pool2 = brew.max_pool(model, conv2, 'pool2', kernel=2, stride=2)

    fc3 = brew.fc(model, pool2, 'fc3', dim_in=50 * 4 * 4, dim_out=500)
    relu3 = brew.relu(model, fc3, 'relu3')

    pred = brew.fc(model, relu3, 'pred', dim_in=500, dim_out=10)
    softmax = brew.softmax(model, pred, 'softmax')

    if save_png:
        graph = net_drawer.GetPydotGraph(model.net)
        graph.write_png("LeNet.png")

    return softmax


def AddModel(model, data, use_lenet=True):
    if use_lenet:
        return AddLeNetModel(model, data)
    else:
        return AddMLPModel(model, data)


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
    optimizer.build_sgd(
        model,
        base_learning_rate=0.1,
        policy="step",
        stepsize=1,
        gamma=0.999,
    )


def AddBookkeepingOperators(model):
    """This adds a few bookkeeping operators that we can inspect later.

    These operators do not affect the training procedure: they only collect
    statistics and prints them to file or to logs.
    """
    # Print basically prints out the content of the blob. to_file=1 routes the
    # printed output to a file. The file is going to be stored under
    #     root_folder/[blob name]
    model.Print('accuracy', [], to_file=1)
    model.Print('loss', [], to_file=1)
    # Summarizes the parameters. Different from Print, Summarize gives some
    # statistics of the parameter, such as mean, std, min and max.
    for param in model.params:
        model.Summarize(param, [], to_file=1)
        model.Summarize(model.param_to_grad[param], [], to_file=1)
    # Now, if we really want to be verbose, we can summarize EVERY blob
    # that the model produces; it is probably not a good idea, because that
    # is going to take time - summarization do not come for free. For this
    # demo, we will only show how to summarize the parameters and their
    # gradients.


if __name__ == "__main__":
    USE_LENET_MODEL = True

    #
    # Train Model
    # Specify the data will be input in NCHW order
    #
    arg_scope = {"order": "NCHW"}

    train_model = model_helper.ModelHelper(name="mnist_train", arg_scope=arg_scope)
    data, label = AddInput(
        train_model,
        batch_size=64,
        db=os.path.join(data_folder, 'mnist-train-nchw-lmdb'),
        db_type='lmdb'
    )
    softmax = AddModel(train_model, data, USE_LENET_MODEL)
    AddTrainingOperators(train_model, softmax, label)
    AddBookkeepingOperators(train_model)

    # visualize the model
    graph = net_drawer.GetPydotGraph(train_model.net.Proto().op, "mnist", rankdir="LR")
    # graph = net_drawer.GetPydotGraphMinimal(
    #     train_model.net.Proto().op, "mnist", rankdir="LR", minimal_dependency=True)
    graph.write_png("process.png")

    #
    # Testing model.
    # We will set the batch size to 100, so that the testing
    #   pass is 100 iterations (10,000 images in total).
    #   For the testing model, we need the data input part, the main AddModel
    #   part, and an accuracy part. Note that init_params is set False because
    #   we will be using the parameters obtained from the train model which will
    #   already be in the workspace.
    #
    test_model = model_helper.ModelHelper(name="mnist_test", arg_scope=arg_scope, init_params=False)
    data, label = AddInput(
        test_model,
        batch_size=100,
        db=os.path.join(data_folder, 'mnist-test-nchw-lmdb'),
        db_type='lmdb'
    )
    softmax = AddModel(test_model, data, USE_LENET_MODEL)
    AddAccuracy(test_model, softmax, label)

    #
    # Deployment model.
    # We simply need the main AddModel part.
    #
    deploy_model = model_helper.ModelHelper(
        name="mnist_deploy", arg_scope=arg_scope, init_params=False)
    AddModel(deploy_model, "data", USE_LENET_MODEL)

    #
    # Training
    #
    workspace.RunNetOnce(train_model.param_init_net)
    workspace.CreateNet(train_model.net, overwrite=True)
    total_iters = 200
    accuracy = np.zeros(total_iters)
    loss = np.zeros(total_iters)

    for i in range(total_iters):
        workspace.RunNet(train_model.net)
        accuracy[i] = workspace.blobs['accuracy']
        loss[i] = workspace.blobs['loss']
        # Check the accuracy and loss every so often
        if i % 25 == 0:
            print("Iter: {}, Loss: {}, Accuracy: {}".format(i, loss[i], accuracy[i]))

    #
    # visualize the data and the results
    #
    plt.figure("Summary of Training")
    plt.title("Summary of Training Run")
    plt.plot(loss, 'b')
    plt.plot(accuracy, 'r')
    plt.xlabel("Iteration")
    plt.legend(('Loss', 'Accuracy'), loc='upper right')

    plt.figure("Training Data")
    plt.title("Training Data Sample")
    data = workspace.FetchBlob('data')
    _ = visualize.NCHW.ShowMultiple(data)

    plt.figure("Softmax Prediction")
    plt.title("Softmax Prediction for the first image above")
    plt.ylabel('Confidence')
    plt.xlabel('Label')
    # Grab and visualize the softmax blob for the batch we just visualized. Since batch size
    #  is 64, the softmax blob contains 64 vectors, one for each image in the batch. To grab
    #  the vector for the first image, we can simply index the fetched softmax blob at zero.
    softmax = workspace.FetchBlob('softmax')
    _ = plt.plot(softmax[0], 'ro')

    # if USE_LENET_MODEL:
    #     plt.figure("Conv1 5th Feature Maps")
    #     plt.title("Conv1 Output Feature Maps for Most Recent Mini-batch")
    #     conv = workspace.FetchBlob('conv1')
    #     conv = conv[:, [5], :, :]  # NCHW, 5th channel
    #     _ = visualize.NCHW.ShowMultiple(conv)
    #
    #     plt.figure("Conv2 5th Feature Maps")
    #     plt.title("Conv2 Output Feature Maps for Most Recent Mini-batch")
    #     conv = workspace.FetchBlob('conv2')
    #     conv = conv[:, [5], :, :]
    #     _ = visualize.NCHW.ShowMultiple(conv)

    #
    # Testing
    #
    workspace.RunNetOnce(test_model.param_init_net)
    workspace.CreateNet(test_model.net, overwrite=True)
    test_accuracy = np.zeros(100)

    for i in range(100):
        workspace.RunNet(test_model.net)
        test_accuracy[i] = workspace.FetchBlob('accuracy')

    plt.figure("Test result")
    plt.title('Accuracy over test batches.')
    plt.plot(test_accuracy, 'r')
    plt.ylim(0.6, 1.2)
    print('test_accuracy: %f' % test_accuracy.mean())
    plt.show()

    #
    # save model
    #
    pe_meta = pe.PredictorExportMeta(
        predict_net=deploy_model.net.Proto(),
        parameters=[str(b) for b in deploy_model.params],
        inputs=["data"],
        outputs=["softmax"],
    )

    pe.save_to_db("minidb", os.path.join(root_folder, "mnist_model.minidb"), pe_meta)
    print("Deploy model saved to: " + root_folder + "/mnist_model.minidb")
# [start-20181001-ben-add] #
    # Save and compare the model with diffrent format
    # Use mobile_exporter's Export function to acquire init_net and predict_net
    init_net, predict_net = me.Export(workspace, deploy_model.net, deploy_model.params)

    full_init_net_out = os.path.join(root_folder, "init_net_out.pb")
    full_predict_net_out = os.path.join(root_folder, "predict_net_out.pb")

    # Simply write the two nets to file
    with open(full_init_net_out, 'wb') as f:
        f.write(init_net.SerializeToString())
    with open(full_predict_net_out, 'wb') as f:
        f.write(predict_net.SerializeToString())
    print("Model saved as " + full_init_net_out + " and " + full_predict_net_out)
# [end-20181001-ben-add] #

    #
    # load model
    #
    blob = workspace.FetchBlob("data")

    plt.figure("Load data")
    plt.title("Batch of Testing Data")
    _ = visualize.NCHW.ShowMultiple(blob)

    # reset the workspace, to make sure the model is actually loaded
    print("The blobs in the workspace before reset: {}".format(workspace.Blobs()))
    workspace.ResetWorkspace(root_folder)
    print("The blobs in the workspace after reset: {}".format(workspace.Blobs()))  # all blobs are destroyed

    # load the predict net and verify the blobs
    predict_net = pe.prepare_prediction_net(os.path.join(root_folder, "mnist_model.minidb"), "minidb")
    print("The blobs in the workspace after loading the model: {}".format(workspace.Blobs()))

    # feed the previously saved data to the loaded model
    workspace.FeedBlob("data", blob)
    workspace.RunNetOnce(predict_net)
    softmax = workspace.FetchBlob("softmax")

    print("Shape of softmax: ", softmax.shape)

    # Quick way to get the top-1 prediction result
    # Squeeze out the unnecessary axis. This returns a 1-D array of length 10
    # Get the prediction and the confidence by finding the maximum value and index of maximum value in preds array
    curr_pred, curr_conf = max(enumerate(softmax[0]), key=operator.itemgetter(1))
    print("Prediction: ", curr_pred)
    print("Confidence: ", curr_conf)

    # the first letter should be predicted correctly
    plt.figure("Previous model")
    plt.title('Prediction for the first image')
    plt.ylabel('Confidence')
    plt.xlabel('Label')
    _ = plt.plot(softmax[0], 'ro')
    plt.show()
