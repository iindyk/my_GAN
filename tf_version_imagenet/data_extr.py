import pickle

f_read = open("/home/iindyk/PycharmProjects/my_GAN/ImageNet/train_data_batch_10", "rb")
data = pickle.load(f_read)
f_read.close()

f_read1 = open("/home/iindyk/PycharmProjects/my_GAN/ImageNet/data.p", "rb")
exist_data = pickle.load(f_read1)
f_read1.close()
train_data = exist_data['train_data']
train_labels = exist_data['train_labels']
#train_data = []
#train_labels = []

labels_to_use = {1, 10, 999}

n = len(data['labels'])

for i in range(n):
    if data['labels'][i] in labels_to_use:
        train_data.append(data['data'][i])
        train_labels.append(data['labels'][i])

data_to_save = {'train_data': train_data, 'train_labels': train_labels}
pickle.dump(data_to_save, open("/home/iindyk/PycharmProjects/my_GAN/ImageNet/data.p", "wb"))