from IPython.display import Image

Image('images/02_network_flowchart.png')

Image('images/02_convolution.png')

# %matplotlib inline
import math
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parameter import Parameter
from torchvision import datasets, transforms
from torch.autograd import Variable

plt.ioff()
cuda = torch.cuda.is_available()

batch_size = 128
test_batch_size = 128
epochs = 4


def array_from_variable(var):
    return var.cpu().data.numpy()


def array_from_tensor(tensor):
    return tensor.cpu().numpy()


def train(epoch, model, optimizer, train_loader):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        prediction = model(data)
        loss = F.cross_entropy(input=prediction, target=target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.data[0]))


def test(model, test_loader):
    model.eval()  # Actually necessary if the model includes layers like Dropout or BatchNorm
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if cuda:
            data, target = data.cuda(), target.cuda()
        # Variable below set to volatile for efficiency, as the model here is used for inference only, no training
        data, target = Variable(data, volatile=True), Variable(target)
        prediction = model(data)
        test_loss += F.cross_entropy(prediction, target, size_average=False).data[0]  # sum up batch loss
        pred_idx = prediction.data.max(1, keepdim=True)[1]  # get the index of the max score
        correct += pred_idx.eq(target.data.view_as(pred_idx)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def plot_classified_sample(dataset_loader, model):
    data_iter = iter(dataset_loader)
    images, labels = data_iter.next()
    images = images[0:9]
    labels = labels.numpy()[0:9]
    if cuda:
        images = images.cuda()
    images = Variable(images, volatile=True)
    pred_scores = model(images)
    predicted = pred_scores.data.max(1, keepdim=True)[1]

    plot_images(array_from_variable(images), cls_true=labels, cls_pred=array_from_tensor(predicted))


def plot_misclassified_sample(dataset_loader, model):
    data_iter = iter(dataset_loader)
    misclassified, prediction, correct_cls = [], [], []
    # Fetch enough samples from the dataset to collect at least 9 misclassified samples
    for images, labels in data_iter:
        if cuda:
            images = images.cuda()
        images = Variable(images, volatile=True)
        pred_scores = model(images)
        predicted = array_from_tensor(pred_scores.data.max(1, keepdim=True)[1]).squeeze()
        incorrect = (predicted != labels.numpy())
        misclassified.extend(array_from_variable(images)[incorrect])
        prediction.extend(predicted[incorrect])
        correct_cls.extend(labels.numpy()[incorrect])
        if len(misclassified) >= 9:
            break

    plot_images(images=misclassified[0:9],
                cls_true=correct_cls[0:9],
                cls_pred=prediction[0:9])


''' Set the dataset loaders'''

kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=batch_size, shuffle=True, **kwargs)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])),
    batch_size=test_batch_size, shuffle=True, **kwargs)

# Count items in datasets
test_data_iter = iter(test_loader)
test_set_size = sum(len(batch[0]) for batch in test_data_iter)
training_data_iter = iter(train_loader)
training_set_size = sum(len(batch[0]) for batch in training_data_iter)

print("Size of:")
print("- Training-set:\t\t{}".format(training_set_size))
print("- Test-set:\t\t{}".format(test_set_size))

# We know that MNIST images are 28 pixels in each dimension.
img_size = 28

# Images are stored in one-dimensional arrays of this length.
img_size_flat = img_size * img_size

# Tuple with height and width of images used to reshape arrays.
img_shape = (img_size, img_size)

# Number of colour channels for the images: 1 channel for gray-scale.
num_channels = 1

# Number of classes, one class for each of 10 digits.
num_classes = 10


def plot_images(images, cls_true, cls_pred=None):
    assert len(images) == len(cls_true) == 9

    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(images[i].reshape(img_shape), cmap='binary')

        # Show true and predicted classes.
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])

        # Show the classes as the label on the x-axis.
        ax.set_xlabel(xlabel)

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()


''' Fetch nine images from the training-set and plot them. Note they can differ from run to run, as the
training-set gets shuffled '''

sample_data_iter = iter(train_loader)
images, labels = sample_data_iter.next()
images = images.numpy()[0:9]
labels = labels.numpy()[0:9]
plot_images(images, labels)

''' Define a sequential model.'''

conv1_kernel_size = 5
conv1_pad = int((conv1_kernel_size - 1) / 2)
assert ((conv1_kernel_size - 1) % 2 == 0)
conv2_kernel_size = 5
conv2_pad = int((conv2_kernel_size - 1) / 2)


# Pytorch sequential model doesn't include a layer for flattening, oddly enough; so we make our own.
class Flatten(nn.Module):

    def forward(self, x):
        return x.view(x.size(0), -1)

    def __repr__(self):
        return 'Flatten()'


sequential_model = nn.Sequential(
    nn.Conv2d(in_channels=1, out_channels=16, kernel_size=conv1_kernel_size, padding=conv1_pad),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),
    nn.Conv2d(in_channels=16, out_channels=36, kernel_size=conv2_kernel_size, padding=conv2_pad),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),
    Flatten(),
    nn.Linear(in_features=1764, out_features=128),  # 36 channels times 7x7 pictures = 1764 input features
    nn.ReLU(),
    nn.Linear(in_features=128, out_features=num_classes)
)

if cuda:
    sequential_model = sequential_model.cuda()

optimizer = optim.Adam(params=sequential_model.parameters(), lr=1e-3)

print('Training sequential model')

for epoch in range(epochs):
    train(epoch, sequential_model, optimizer, train_loader)
    test(sequential_model, test_loader)

plot_classified_sample(train_loader, sequential_model)

plot_misclassified_sample(test_loader, sequential_model)


''' Define a functional model.'''


class FunctionalModel(nn.Module):
    def __init__(self):
        super(FunctionalModel, self).__init__()

        def get_conv_weight_and_bias(n_input, n_output, k1, k2):
            w = Parameter(torch.Tensor(n_output, n_input, k1, k2))
            stdv = 1. / math.sqrt(n_input)
            nn.init.uniform(w, -stdv, stdv)
            b = Parameter(torch.Tensor(n_output))
            nn.init.uniform(b, -stdv, stdv)
            return w, b

        def get_linear_weight_and_bias(n_input, n_output):
            w = Parameter(torch.Tensor(n_output, n_input))
            stdv = 1. / math.sqrt(w.size(1))
            nn.init.uniform(w, -stdv, stdv)
            b = Parameter(torch.Tensor(n_output))
            nn.init.uniform(b, -stdv, stdv)
            return w, b

        conv1_kernel_size = 5
        assert ((conv1_kernel_size - 1) % 2 == 0)
        self.conv1_pad = int((conv1_kernel_size - 1) / 2)
        conv2_kernel_size = 5
        assert ((conv2_kernel_size - 1) % 2 == 0)
        self.conv2_pad = int((conv2_kernel_size - 1) / 2)

        self.conv1_w, self.conv1_b = get_conv_weight_and_bias(n_input=1, n_output=16, k1=conv1_kernel_size,
                                                              k2=conv1_kernel_size)
        self.conv2_w, self.conv2_b = get_conv_weight_and_bias(n_input=16, n_output=36, k1=conv2_kernel_size,
                                                              k2=conv2_kernel_size)
        self.fc1_w, self.fc1_b = get_linear_weight_and_bias(n_input=1764, n_output=128)
        self.fc2_w, self.fc2_b = get_linear_weight_and_bias(n_input=128, n_output=num_classes)

    def forward(self, x):
        x = F.conv2d(input=x, weight=self.conv1_w, bias=self.conv1_b, padding=self.conv1_pad)
        x = F.max_pool2d(F.relu(x), 2)
        x = F.conv2d(input=x, weight=self.conv2_w, bias=self.conv2_b, padding=self.conv2_pad)
        x = F.max_pool2d(F.relu(x), 2)
        x = x.view(x.size(0), -1)  # Flatten but keep the same batch size
        x = F.linear(input=x, weight=self.fc1_w, bias=self.fc1_b)
        x = F.relu(x)
        x = F.linear(input=x, weight=self.fc2_w, bias=self.fc2_b)
        return x


functional_model = FunctionalModel()
if cuda:
    functional_model.cuda()

optimizer = optim.Adam(params=functional_model.parameters(), lr=1e-3)

print('Training functional model')

for epoch in range(epochs):
    train(epoch, functional_model, optimizer, train_loader)
    test(functional_model, test_loader)

# Plot 9 images correctly classified, along with their classification
plot_classified_sample(train_loader, functional_model)

# Plot 9 images that have been misclassified, along with their (incorrect) classification and correct class
plot_misclassified_sample(test_loader, functional_model)

saved_model_path = 'model.p'
torch.save(functional_model.state_dict(), saved_model_path)

del functional_model

loaded_model = FunctionalModel()
if cuda:
    loaded_model = loaded_model.cuda()
state_dict = torch.load(saved_model_path)
loaded_model.load_state_dict(state_dict)
loaded_model.eval()

plot_classified_sample(train_loader, loaded_model)


def plot_conv_weights(weights, input_channel=0):
    # Get the lowest and highest values for the weights.
    # This is used to correct the colour intensity across
    # the images so they can be compared with each other.
    w_min = np.min(weights)
    w_max = np.max(weights)

    # Number of filters used in the conv. layer.
    num_filters = weights.shape[0]

    # Number of grids to plot.
    # Rounded-up, square-root of the number of filters.
    num_grids = math.ceil(math.sqrt(num_filters))

    # Create figure with a grid of sub-plots.
    fig, axes = plt.subplots(num_grids, num_grids)

    # Plot all the filter-weights.
    for i, ax in enumerate(axes.flat):
        # Only plot the valid filter-weights.
        if i < num_filters:
            # Get the weights for the i'th filter of the input channel.
            # See new_conv_layer() for details on the format
            # of this 4-dim tensor.
            img = weights[i, input_channel, :, :]
            # img = weights[:, :, input_channel, i]

            # Plot image.
            ax.imshow(img, vmin=w_min, vmax=w_max,
                      interpolation='nearest', cmap='seismic')

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()


print(sequential_model)

plot_conv_weights(sequential_model[0].weight.data.cpu().numpy())
plot_conv_weights(loaded_model.conv2_w.data.cpu().numpy())


def plot_conv_output(values):
    # Number of filters used in the conv. layer.
    num_filters = values.shape[1]

    # Number of grids to plot.
    # Rounded-up, square-root of the number of filters.
    num_grids = math.ceil(math.sqrt(num_filters))

    # Create figure with a grid of sub-plots.
    fig, axes = plt.subplots(num_grids, num_grids)

    # Plot the output images of all the filters.
    for i, ax in enumerate(axes.flat):
        # Only plot the images for valid filters.
        if i < num_filters:
            # Get the output image of using the i'th filter.
            img = values[0, i, :, :]

            # Plot image.
            ax.imshow(img, interpolation='nearest', cmap='binary')

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()


# Change [0:1] to [0:3] to access the output of the second convolutional layer instead
sub_model = nn.Sequential(*list(sequential_model.children())[0:1])

print(sub_model)

sample_data_iter = iter(train_loader)
images, labels = sample_data_iter.next()


def plot_image(image):
    plt.imshow(image.reshape(img_shape),
               interpolation='nearest',
               cmap='binary')

    plt.show()


plot_image(images[0].numpy())

if cuda:
    images = images.cuda()
images = images[0:1]
images = Variable(images, volatile=True)
conv1_output = sub_model(images)
plot_conv_output(conv1_output.data.cpu().numpy())
