# convert_cifar_for_hashing
In terms of hashing and quantization, people usually split the cifar10 dataset to a training dataset containg 59000 images and a testing dataset contrain 1000 images. And Caffe is very popular with the Deep Learning is getting hotter.<br>
So We split the cifar10 dataset and save it to the LMDB which is the default dataset storage format of Caffe.<br>
In addition, We noticed that people want to use the AlexNet to train Cifar10 for hashing and int this case we need resize the image.
Usage: <br>
python convert_cifar_for_hashing.py [--resize=256] path-of-cifar10-batches-py [the-path-of-lmdb]<br>
Note:remember add the path of 'caffe/python' to the environment variable PYTHONPATH
