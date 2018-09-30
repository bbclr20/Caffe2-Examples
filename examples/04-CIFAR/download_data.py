#!/usr/bin/python3
from matplotlib import pyplot as plt
import numpy as np
import os
import lmdb
from imageio import imread
from caffe2.proto import caffe2_pb2
from caffe2.python import core
import glob
from random import shuffle
import requests
import tarfile

core.GlobalInit(["caffe2", "--caffe2_log_level=0"])
print("Necessities imported!")

#
# Download data
#
data_folder = os.path.join("tutorial_data", "cifar10")
root_folder = os.path.join("tutorial_files", "tutorial_cifar10")

url = "http://pjreddie.com/media/files/cifar.tgz"
filename = url.split("/")[-1]
download_path = os.path.join(data_folder, filename)

if not os.path.isdir(data_folder):
    os.makedirs(data_folder)

if not os.path.exists(download_path):
    r = requests.get(url, stream=True)

    print("Downloading... {} to {}".format(url, download_path))
    open(download_path, 'wb').write(r.content)
    print("Finished downloading...")

    print('Extracting images from tarball...')
    tar = tarfile.open(download_path, 'r')
    for item in tar:
        tar.extract(item, data_folder)
    print("Completed download and extraction!")
else:
    print("Image directory already exists. Moving on...")

#
# Plot images
#
sample_imgs = glob.glob(os.path.join(data_folder, "cifar", "train") + '/*.png')[:20]
plt.subplots(4, 5, figsize=(10, 10))
plt.tight_layout()
for i in range(20):
    ax = plt.subplot(4, 5, i+1)
    ax.set_title(sample_imgs[i].split("_")[-1].split(".")[0])
    ax.imshow(imread(sample_imgs[i]).astype(np.uint8))
    ax.axis("off")
plt.show()

#
# Set the paths
#
# Paths to train and test directories
training_dir_path = os.path.join("tutorial_data", "cifar10", "cifar", "train")
testing_dir_path = os.path.join("tutorial_data", "cifar10", "cifar", "test")

# Paths to label files
training_labels_path = os.path.join("tutorial_data", "cifar10", "training_dictionary.txt")
validation_labels_path = os.path.join("tutorial_data", "cifar10", "validation_dictionary.txt")
testing_labels_path = os.path.join("tutorial_data", "cifar10", "testing_dictionary.txt")

# Paths to LMDBs
training_lmdb_path = os.path.join("tutorial_data", "cifar10", "training_lmdb")
validation_lmdb_path = os.path.join("tutorial_data", "cifar10", "validation_lmdb")
testing_lmdb_path = os.path.join("tutorial_data", "cifar10", "testing_lmdb")

# Path to labels.txt
labels_path = os.path.join("tutorial_data", "cifar10", "cifar", "labels.txt")

with open(labels_path, "r") as labels_handler:
    classes = {}
    i = 0
    lines = labels_handler.readlines()
    for line in sorted(lines):
        line = line.rstrip()
        classes[line] = i
        i += 1
print("classes:", classes)

#
# Create training, validation, and testing label files
#
with open(training_labels_path, "w") as training_labels_handler:
    with open(validation_labels_path, "w") as validation_labels_handler:
        # Write our training label files using the training images
        i = 0
        validation_count = 6000
        imgs = glob.glob(training_dir_path + '/*.png')
        shuffle(imgs)  # shuffle array

        for img in imgs:
            # Write first 6,000 image paths, followed by their integer label, to the validation label files
            if i < validation_count:
                validation_labels_handler.write(img + ' ' + str(classes[img.split('_')[-1].split('.')[0]]) + '\n')
            # Write the remaining to the training label files
            else:
                training_labels_handler.write(img + ' ' + str(classes[img.split('_')[-1].split('.')[0]]) + '\n')
            i += 1
        print("Finished writing training and validation label files")

with open(testing_labels_path, "w") as testing_labels_handler:
    # Write our testing label files using the testing images
    for img in glob.glob(testing_dir_path + '/*.png'):
        testing_labels_handler.write(img + ' ' + str(classes[img.split('_')[-1].split('.')[0]]) + '\n')
    print("Finished writing testing label files")


#
# Write to db
#
def write_lmdb(labels_file_path, lmdb_path):
    with open(labels_file_path, "r") as labels_handler:
        print(">>> Write database...")
        LMDB_MAP_SIZE = 1 << 40  # ????
        print("LMDB_MAP_SIZE", LMDB_MAP_SIZE)

        env = lmdb.open(lmdb_path, map_size=LMDB_MAP_SIZE)

        with env.begin(write=True) as txn:
            count = 0
            for line in labels_handler.readlines():
                line = line.rstrip()
                im_path = line.split()[0]
                im_label = int(line.split()[1])

                # read in image (as RGB)
                img_data = imread(im_path).astype(np.float32)

                # convert to BGR
                img_data = img_data[:, :, (2, 1, 0)]
                img_data = img_data / 256.0

                # HWC -> CHW (N gets added in AddInput function)
                img_data = np.transpose(img_data, (2, 0, 1))

                # Create TensorProtos
                tensor_protos = caffe2_pb2.TensorProtos()

                img_tensor = tensor_protos.protos.add()
                img_tensor.dims.extend(img_data.shape)
                img_tensor.data_type = 1
                flatten_img = img_data.reshape(np.prod(img_data.shape))
                img_tensor.float_data.extend(flatten_img)

                label_tensor = tensor_protos.protos.add()
                label_tensor.data_type = 2
                label_tensor.int32_data.append(im_label)

                # write to db
                txn.put(
                    "{}".format(count).encode("ascii"),  # key
                    tensor_protos.SerializeToString()    # value
                )
                if count % 1000 == 0:
                    print("Inserted {} rows".format(count))
                count = count + 1
        print("Inserted {} rows".format(count))
        print("\nLMDB saved at " + lmdb_path)


#
# Call function to write our LMDBs
#
if not os.path.exists(training_lmdb_path):
    print("Writing training LMDB")
    write_lmdb(training_labels_path, training_lmdb_path)
else:
    print(training_lmdb_path, "already exists!")

if not os.path.exists(validation_lmdb_path):
    print("Writing validation LMDB")
    write_lmdb(validation_labels_path, validation_lmdb_path)
else:
    print(validation_lmdb_path, "already exists!")

if not os.path.exists(testing_lmdb_path):
    print("Writing testing LMDB")
    write_lmdb(testing_labels_path, testing_lmdb_path)
else:
    print(testing_lmdb_path, "already exists!")
