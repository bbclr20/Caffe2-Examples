import numpy as np
import skimage.io
import skimage.transform
from matplotlib import pyplot as plt
import os
from caffe2.python import workspace
import operator
import urllib

#
# Download caffe2 models from github
# git clone https://github.com/caffe2/models to ~/caffe2/caffe2/python/
#
CAFFE_MODELS = "~/caffe2/caffe2/python/models"

# Format below is the model's: <folder, INIT_NET, predict_net, mean, input image size>
# You can switch 'squeezenet' out with 'bvlc_alexnet', 'bvlc_googlenet' or others that you have downloaded
MODEL = ("squeezenet", "init_net.pb", "predict_net.pb", "ilsvrc_2012_mean.npy", 227)

# codes - these help decypher the output and source from a list from ImageNet's object codes
#   to provide an result like "tabby cat" or "lemon" depending on what's in the picture
#   you submit to the CNN.
codes = "https://gist.githubusercontent.com/aaronmarkham/cd3a6b6ac071eca6f7b4a6e40e6038aa/raw/9edb4038a37da6b5a44c3b5bc52e448ff09bfe5b/alexnet_codes"
print("Config set!")

CAFFE_MODELS = os.path.expanduser(CAFFE_MODELS)

# mean can be 128 or custom based on the model
# gives better results to remove the colors found in all of the training images
MEAN_FILE = os.path.join(CAFFE_MODELS, MODEL[0], MODEL[3])
if not os.path.exists(MEAN_FILE):
    print("No mean file found!")
    mean = 128
else:
    print("Mean file found!")
    mean = np.load(MEAN_FILE).mean(1).mean(1)
    mean = mean[:, np.newaxis, np.newaxis]
print("mean was set to: \n{}\n".format(mean))

#
# Set init_net and predict_net
#
INPUT_IMAGE_SIZE = MODEL[4]
INIT_NET = os.path.join(CAFFE_MODELS, MODEL[0], MODEL[1])
PREDICT_NET = os.path.join(CAFFE_MODELS, MODEL[0], MODEL[2])

if not os.path.exists(INIT_NET):
    print("WARNING: " + INIT_NET + " not found!")
else:
    if not os.path.exists(PREDICT_NET):
        print("WARNING: " + PREDICT_NET + " not found!")
    else:
        print("INIT_NET and PREDICT_NET are found!")


def crop_center(img, crop_x, crop_y):
    """Function to crop the center cropX x cropY pixels from the input image"""
    y, x, c = img.shape
    start_x = x//2-(crop_x//2)
    start_y = y//2-(crop_y//2)
    return img[start_y:start_y+crop_y, start_x:start_x+crop_x]


def rescale(img, input_height, input_width):
    """Function to rescale the input image to the desired height and/or width.
    This function will preserve the aspect ratio of the original image."""
    aspect = img.shape[1]/float(img.shape[0])
    if aspect > 1:
        res = int(aspect * input_height)
        img_scaled = skimage.transform.resize(
            img, (input_width, res), mode="constant", anti_aliasing=False)
    if aspect < 1:
        res = int(input_width/aspect)
        img_scaled = skimage.transform.resize(
            img, (res, input_height), mode="constant", anti_aliasing=False)
    if aspect == 1:
        img_scaled = skimage.transform.resize(
            img, (input_width, input_height), mode="constant", anti_aliasing=False)
    return img_scaled


if __name__ == "__main__":
    IMAGE_LOCATION = "../../images/flower.jpg"

    img = skimage.img_as_float(skimage.io.imread(IMAGE_LOCATION))
    img = img.astype(np.float32)

    print("Original Image Shape: ", img.shape)
    plt.figure("Flower")
    ax = plt.subplot(1, 3, 1)
    ax.set_title("Original image")
    ax.imshow(img)

    # Rescale the image and keep the aspect ratio
    img = rescale(img, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE)
    print("Image Shape after rescaling: ", img.shape)
    ax = plt.subplot(1, 3, 2)
    ax.set_title("Rescaled image")
    ax.imshow(img)

    # Crop the center 227x227 pixels of the image
    img = crop_center(img, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE)
    print("Image Shape after cropping: ", img.shape)
    ax = plt.subplot(1, 3, 3)
    ax.set_title("Center cropped")
    ax.imshow(img)
    plt.show()

    #
    # Caffe2 use NCHW dataset and BGR data
    #
    # Switch to CHW (HWC --> CHW)
    img = img.swapaxes(1, 2).swapaxes(0, 1)
    print("CHW Image Shape: ", img.shape)

    plt.figure("RGB image")
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.imshow(img[i])
        plt.axis("off")
        plt.title("RGB channel %d" % (i+1))
    plt.show()

    # switch to BGR (RGB --> BGR)
    img = img[(2, 1, 0), :, :]
    img = img * 255 - mean
    img = img[np.newaxis, :, :, :].astype(np.float32)

    print("NCHW image (ready to be used as input): ", img.shape)

    #
    # Load pre-trained model
    #
    with open(INIT_NET, "rb") as f:
        init_net = f.read()
    with open(PREDICT_NET, "rb") as f:
        predict_net = f.read()

    p = workspace.Predictor(init_net, predict_net)
    results = p.run({'data': img})

    results = np.asarray(results)
    print("results shape: ", results.shape)

    # Quick way to get the top-1 prediction result
    # Squeeze out the unnecessary axis. This returns a 1-D array of length 1000
    preds = np.squeeze(results)
    curr_pred, curr_conf = max(enumerate(preds), key=operator.itemgetter(1))
    print("Prediction: ", curr_pred)
    print("Confidence: ", curr_conf)

    #
    # The rest of this is digging through the results
    #
    results = np.delete(results, 1)
    index = 0
    highest = 0
    arr = np.empty((0, 2), dtype=object)
    arr[:, 0] = int(10)
    arr[:, 1:] = float(10)
    for i, r in enumerate(results):
        i = i + 1
        arr = np.append(arr, np.array([[i, r]]), axis=0)
        if r > highest:
            highest = r
            index = i

    # top N results
    N = 5
    topN = sorted(arr, key=lambda x: x[1], reverse=True)[:N]
    print("Raw top {} results: {}".format(N, topN))
    topN_inds = [int(x[0]) for x in topN]
    print("Top {} classes in order: {}".format(N, topN_inds))

    #
    # Download and parse the file of classes
    #
    response = urllib.request.urlopen(codes)

    classes = {}
    for line in response:
        line = line.decode("utf-8")
        try:
            code, result = line.split(":")
            classes[code.strip()] = result.strip().replace("'", "")
        except:
            pass

    # For each of the top-N results, associate the integer result with an actual class
    for n in topN:
        print("Model predicts '{}' with {}% confidence".format(
            classes[str(int(n[0]))], float("{0:.2f}".format(n[1] * 100))))

    #
    # Test the network with other images
    #
    images = ["../../images/cowboy-hat.jpg",
              "../../images/cell-tower.jpg",
              "../../images/Ducreux.jpg",
              "../../images/pretzel.jpg",
              "../../images/orangutan.jpg",
              "../../images/aircraft-carrier.jpg",
              "../../images/cat.jpg"]

    NCHW_batch = np.zeros((len(images), 3, 227, 227))
    print("Batch Shape: ", NCHW_batch.shape)

    for i, curr_img in enumerate(images):
        img = skimage.img_as_float(skimage.io.imread(curr_img)).astype(np.float32)
        img = rescale(img, 227, 227)
        img = crop_center(img, 227, 227)
        img = img.swapaxes(1, 2).swapaxes(0, 1)
        img = img[(2, 1, 0), :, :]
        img = img * 255 - mean
        NCHW_batch[i] = img

    print("NCHW image (ready to be used as input): ", NCHW_batch.shape)

    # Run the net on the batch
    results = p.run([NCHW_batch.astype(np.float32)])

    results = np.asarray(results)
    preds = np.squeeze(results)
    print("Squeezed Predictions Shape, with batch size {}: {}".format(len(images),preds.shape))

    for i, pred in enumerate(preds):
        print("Results for: '{}'".format(images[i]))
        curr_pred, curr_conf = max(enumerate(pred), key=operator.itemgetter(1))
        print("\tPrediction: ", curr_pred)
        print("\tClass Name: ", classes[str(int(curr_pred))])
        print("\tConfidence: ", curr_conf)
