import csv as csv
import numpy as np
from PIL import Image
import os
import random
import theano
import theano.tensor as T
import numpy

class sampler(object):
    
    def __init__(self):
        self.labels_file = os.getcwd() + '/data_grey/trainLabels.csv'
        self.train_img_dir = os.getcwd() + '/data_grey/train/'
        self.test_img_dir = os.getcwd() + '/data_grey/test/'
        self.data_x = []
        self.data_y = []

    def sample_files(self):
        np.random.shuffle(self.data)

        zeros = self.data[0::,1] == '0'
        ones = self.data[0::,1] == '1'
        twos = self.data[0::,1] == '2'
        threes = self.data[0::,1] == '3'
        fours = self.data[0::,1] == '4'

        zero_data = self.data[zeros,][0:self.class_counts[0],]
        one_data = self.data[ones,][0:self.class_counts[1],]
        two_data = self.data[twos,][0:self.class_counts[2],]
        three_data = self.data[threes,][0:self.class_counts[3],]
        four_data = self.data[fours,][0:self.class_counts[4],]

        self.sample = np.concatenate((zero_data, one_data, two_data, 
                                        three_data, four_data), axis=0)

    def get_pixel_data(self, file):
        #meh
        im = Image.open(file)
        pixels = list(im.getdata())
        pixel_vals = []
        for pixel in pixels: pixel_vals.append(pixel[0])
        pixel_vals = np.array(pixel_vals)
        pix_mean = pixel_vals.mean()
        pixel_vals_norm = np.multiply(1/pix_mean, pixel_vals) - 1

        return pixel_vals_norm

    def main(self, class_counts, bin=False):
        self.class_counts = class_counts
    
        csv_file_obj = csv.reader(open(self.labels_file, 'rb'))
        header = csv_file_obj.next()

        self.data=[]
        for row in csv_file_obj:
            self.data.append(row)
        self.data = np.array(self.data)

        self.sample_files()

        #self.data_x = []
        #self.data_y = []

        for i in xrange(self.sample.shape[0]):
            file_name = self.train_img_dir + self.sample[i,0] + '.jpeg'
            file_data = self.get_pixel_data(file_name)
            file_data = np.array(file_data)
            self.data_x.append(file_data)
            self.data_y.append(self.sample[i,1])

        if bin is True:
            self.data_y = [int(int(item)>0) for item in self.data_y]

def load_data(class_count):
    bin=False
    class_counts = [class_count] * 5

    load_obj = sampler()
    load_obj.main(class_counts,bin)

    train_size =  int(class_count*0.7)
    xval_size = int((class_count - train_size)/2)
    test_size = class_count - train_size - xval_size

    train_ids = [range(x*class_count,x*class_count+train_size) for x in range(5)]
    xval_ids=[range(x*class_count+train_size,x*class_count+train_size+xval_size) for x in range(5)]
    test_ids = [range(x*class_count-test_size, x*class_count) for x in range(1,6)]

    train_vec = flatten_lists(train_ids)
    xval_vec = flatten_lists(xval_ids)
    test_vec = flatten_lists(test_ids)

    X_train = np.array([load_obj.data_x[index] for index in train_vec])
    y_train = np.array([load_obj.data_y[index] for index in train_vec])

    deck = list(zip(X_train, y_train))
    random.shuffle(deck)
    X_train, y_train = zip(*deck)
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    X_valid = np.array([load_obj.data_x[index] for index in xval_vec])
    y_valid = np.array([load_obj.data_y[index] for index in xval_vec])

    X_test = np.array([load_obj.data_x[index] for index in test_vec])
    y_test = np.array([load_obj.data_y[index] for index in test_vec])

    return [[X_train, y_train.astype(np.float)], 
            [X_valid, y_valid.astype(np.float)], 
            [X_test, y_test.astype(np.float)]]

def load_data_bin(class_count):
    bin=True
    class_counts = [class_count] * 5
    class_counts[0] *= 4

    load_obj = sampler()
    load_obj.main(class_counts,bin)

    train_size = int(class_count*5.6)
    xval_size = int((class_count*8 - train_size)/2)
    test_size = class_count*8-train_size-xval_size

    deck = list(zip(load_obj.data_x, load_obj.data_y))
    random.shuffle(deck)
    x_data, y_data = zip(*deck)

    X_train = np.array(x_data[0:train_size])
    y_train = np.array(y_data[0:train_size])

    X_valid = np.array(x_data[train_size:(train_size+xval_size)])
    y_valid = np.array(y_data[train_size:(train_size+xval_size)])

    X_test = np.array(x_data[-test_size:])
    y_test = np.array(y_data[-test_size:])

    train_set = [X_train,y_train]
    valid_set = [X_valid, y_valid]
    test_set = [X_test, y_test]

    print y_train.mean(), y_valid.mean(), y_test.mean()

    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval

def flatten_lists(lists):
    return [item for sublist in lists for item in sublist]
