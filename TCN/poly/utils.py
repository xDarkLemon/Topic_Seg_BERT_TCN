from scipy.io import loadmat
import torch
import numpy as np
import pickle


def data_generator(dataset):
    if dataset == "JSB":
        print('loading JSB data...')
        data = loadmat('./TCN/poly/mdata/JSB_Chorales.mat')
    
    elif dataset == 'daT':
        print('loading daT data ...')
        train_data = pickle.load(open('./TCN/poly/mdata/daT_train_data.pkl','rb'))
        test_data = pickle.load(open('./TCN/poly/mdata/daT_test_data.pkl','rb'))
        dev_data = pickle.load(open('./TCN/poly/mdata/daT_dev_data.pkl','rb'))

        train_labels = pickle.load(open('./TCN/poly/mdata/daT_train_labels.pkl','rb'))
        test_labels = pickle.load(open('./TCN/poly/mdata/daT_test_labels.pkl','rb'))
        dev_labels = pickle.load(open('./TCN/poly/mdata/daT_dev_labels.pkl','rb'))


        train_data = [torch.Tensor(data.astype(np.float64)) for data in train_data]
        test_data = [torch.Tensor(data.astype(np.float64)) for data in test_data]
        dev_data = [torch.Tensor(data.astype(np.float64)) for data in dev_data]
        train_labels = [torch.Tensor(label).long() for label in train_labels]
        test_labels = [torch.Tensor(label).long() for label in test_labels]
        dev_labels = [torch.Tensor(label).long() for label in dev_labels]

        return train_data,test_data,dev_data,train_labels,test_labels,dev_labels
        

    X_train = data['traindata'][0]
    X_valid = data['validdata'][0]
    X_test = data['testdata'][0]

    for data in [X_train, X_valid, X_test]:
        for i in range(len(data)):
            data[i] = torch.Tensor(data[i].astype(np.float64))

    return X_train, X_valid, X_test

if __name__ == '__main__':
    train_data,test_data,dev_data,train_labels,test_labels,dev_labels = data_generator('DAct')
    print(train_data[0])
    print('dev_data[0].shape:',dev_data[0].shape)
    #print('train_data.shape:',train_data.shape)
    print('type(labels[0]):',type(train_labels[0]))
    print('labels[0]:',train_labels[0])
    print('labels[0].shape:',dev_labels[0].shape)
