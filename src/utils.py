import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn
import ipdb

def to_var(x):
    """ Make a tensor cuda-erized and requires gradient """
    return to_cuda(x).requires_grad_()

def to_cuda(x):
    """ Cuda-erize a tensor """
    if torch.cuda.is_available():
        x = x.cuda()
    return x

def get_data(BATCH_SIZE=100):
    """ Load data for binared MNIST """
    torch.manual_seed(3435)

    # Download our data
    train_dataset = datasets.MNIST(root='../data/',
                                    train=True,
                                    transform=transforms.ToTensor(),
                                    download=True)

    test_dataset = datasets.MNIST(root='../data/',
                                   train=False,
                                   transform=transforms.ToTensor())

    # Use greyscale values as sampling probabilities to get back to [0,1]
    train_img = torch.stack([torch.bernoulli(d[0]) for d in train_dataset])
    train_label = torch.LongTensor([d[1] for d in train_dataset])

    test_img = torch.stack([torch.bernoulli(d[0]) for d in test_dataset])
    test_label = torch.LongTensor([d[1] for d in test_dataset])

    # MNIST has no official train dataset so use last 10000 as validation
    val_img = train_img[-10000:].clone()
    val_label = train_label[-10000:].clone()

    train_img = train_img[:-10000]
    train_label = train_label[:-10000]

    # Create data loaders
    train = torch.utils.data.TensorDataset(train_img, train_label)
    val = torch.utils.data.TensorDataset(val_img, val_label)
    test = torch.utils.data.TensorDataset(test_img, test_label)

    train_iter = torch.utils.data.DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
    val_iter = torch.utils.data.DataLoader(val, batch_size=BATCH_SIZE, shuffle=True)
    test_iter = torch.utils.data.DataLoader(test, batch_size=BATCH_SIZE, shuffle=True)

    return train_iter, val_iter, test_iter




"""
    generate 2d gaussian around a circle
"""
class data_generator(object):
    def __init__(self):

        n = 8
        radius = 2
        std = 0.02
        delta_theta = 2*np.pi / n

        centers_x = []
        centers_y = []
        for i in range(n):
            centers_x.append(radius*np.cos(i*delta_theta))
            centers_y.append(radius*np.sin(i*delta_theta))

        centers_x = np.expand_dims(np.array(centers_x), 1)
        centers_y = np.expand_dims(np.array(centers_y), 1)

        p = [1./n for _ in range(n)]

        self.p = p
        self.size = 2
        self.n = n
        self.std = std
        self.centers = np.concatenate([centers_x, centers_y], 1)

    # switch to random distribution (harder)
    def random_distribution(self, p=None):
        if p is None:
            p = [np.random.uniform() for _ in range(self.n)]
            p = p / np.sum(p)
        self.p = p

    # switch to uniform distribution
    def uniform_distribution(self):
        p = [1./self.n for _ in range(self.n)]
        self.p = p

    def sample(self, N):
        n = self.n
        std = self.std
        centers = self.centers

        ith_center = np.random.choice(n, N,p=self.p)
        sample_centers = centers[ith_center, :]
        sample_points = np.random.normal(loc=sample_centers, scale=std)
        return sample_points.astype('float32')


def plot(points):
    plt.scatter(points[:, 0], points[:, 1], c=[0.3 for i in range(1000)], alpha=0.5)
    plt.show()
    plt.close()

def main():
    gen = data_generator()
    sample_points = gen.sample(1000)
    plot(sample_points)

if __name__ == '__main__':
    main()