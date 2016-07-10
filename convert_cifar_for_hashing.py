import cPickle
import numpy as np
import lmdb
import sys
import os.path
from caffe.proto.caffe_pb2 import Datum

def load_cifar10(dataset_path):
    """
    loading the cifar10 dataset
    :param dataset_path:
    :return:
    """
    train_data=np.zeros((50000,3072),dtype=np.uint8)
    train_label=[]

    for batch_id in range(1,6):
        batch_file_name=dataset_path+'/data_batch_'+str(batch_id)
        batch_file=open(batch_file_name,'rb')
        batch_data=cPickle.load(batch_file)
        train_data[(batch_id-1)*10000:batch_id*10000,:]=batch_data['data']
        train_label=train_label+batch_data['labels']

    test_file=open(dataset_path+'/test_batch','rb')
    test_data_dict=cPickle.load(test_file)
    test_label=test_data_dict['labels']
    test_data=test_data_dict['data']

    return split_cifar10_forNN(train_data,np.array(train_label),test_data,np.array(test_label))


def split_cifar10_forNN(train_data,train_label,test_data,test_label):
    """
        In terms of hashing and quantization, people usually resplit cifar10 dataset to 59000 training images
        and 1000 testing images
    :param train_data:
    :param test_data:
    :return:
    """
    splited_train_data=np.zeros((59000,3072),dtype=np.uint8)
    splited_train_label=np.zeros((59000,),dtype=np.int64)
    splited_test_data=np.zeros((1000,3072),dtype=np.uint8)
    splited_test_label=np.zeros((1000,),dtype=np.int64)

    splited_train_data[0:50000,:]=train_data
    splited_train_label[0:50000]=train_label
    for label in range(0,10):
        data_idx=(np.array(test_label)==label).nonzero()[0]
        test_idx=data_idx[0:100]
        train_idx=data_idx[100:]
        splited_train_data[50000+label*900:50000+(label+1)*900,:]=test_data[train_idx,:]
        splited_train_label[50000+label*900:50000+(label+1)*900]=test_label[train_idx]
        splited_test_data[label*100:(label+1)*100,:]=test_data[test_idx,:]
        splited_test_label[label*100:(label+1)*100]=test_label[test_idx]

    train_data_range = range(splited_train_data.shape[0])
    np.random.shuffle(train_data_range)

    return (splited_train_data[train_data_range,:],splited_train_label[train_data_range],
            splited_test_data,splited_test_label)
# def calc_img_mean(image_path):
#
#     batch_count=59000/256
#     img_sum=np.zeros((3,256,256))
#     for batch_idx in range(0,batch_count):
#         print 'batch idx:%d' %batch_idx
#         file_name=image_path+('/cifar10_train_%04d.hkl' %batch_idx)
#         batch_data=hkl.load(file_name)
#         img_sum+=np.sum(batch_data,0)
#     img_mean=img_sum/59000
#     np.save(image_path+'/img_mean.npy',img_mean)

def build_dataset(img_data,label,dataset_path):
    """
    build the lmdb-format training dataset
    :param train_data:
    :param train_label:
    :param lmdb_filename:
    :return:
    """
    data_size=img_data.shape[0]
    img_width=32
    img_height=32
    img_channel=3

    # reshape to img
    img_data=np.reshape(img_data,(data_size,3,32,32))

    # convert RGB to BGR
    img_data=img_data[:,::-1,:,:]

    # open the lmdb
    map_size=img_data.nbytes*10
    env=lmdb.open(dataset_path,map_size=map_size)
    with env.begin(write=True) as txn:

        #txn is a Transaction
        for data_idx in range(data_size):
            img=img_data[data_idx,:,:,:]
            if (data_idx+1)%10000==0:
                print "[msg]%d images have been written" % data_idx
            datum=Datum()
            datum.channels=img_channel
            datum.height=img_height
            datum.width=img_width
            datum.label=int(label[data_idx])

            datum.data=img.tobytes()
            str_id='{:08}'.format(data_idx)
            txn.put(str_id,datum.SerializeToString())


if __name__=='__main__':

    if len(sys.argv)<2:
        print "build the cifar10 lmdb dataset\nUsage:\npython convert_cifar_for_hashing.py "\
        "path-of-cifar10-batches-py [the-path-of-lmdb]\nNote:remember add the path of pycaffe to the PYTHONPATH "
        sys.exit(1)
    (train_data,train_label,test_data,test_label)= load_cifar10(sys.argv[1])

    if len(sys.argv)==3:
        base_dir=sys.argv[2]
    else:
        base_dir=os.getcwd()

    train_path=os.path.join(base_dir,'cifar10_train_lmdb')
    test_path=os.path.join(base_dir,'cifar10_test_lmdb')
    if os.path.exists(train_path)==False:
        os.mkdir(train_path)
    if os.path.exists(test_path)==False:
        os.mkdir(test_path)

    build_dataset(train_data,train_label,train_path)
    build_dataset(test_data,test_label,test_path)
    print  'Done!'
