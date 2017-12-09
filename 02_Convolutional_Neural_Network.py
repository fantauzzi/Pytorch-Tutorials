from IPython.display import Image

Image('images/02_network_flowchart.png')

Image('images/02_convolution.png')

# %matplotlib inline
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.optim as optim
from torch.autograd import Variable
import math

plt.ioff()
cuda = torch.cuda.is_available()

batch_size = 128
test_batch_size = 128
epochs = 1


def train(epoch, model, train_loader):
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


def test(model):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        prediction = model(data)
        test_loss += F.cross_entropy(prediction, target, size_average=False).data[0]  # sum up batch loss
        pred = prediction.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def array_from_variable(var):
    return var.cpu().data.numpy()


def array_from_tensor(tensor):
    return tensor.cpu().numpy()


# TODO check if you should load the TF dataset instead
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=batch_size, shuffle=True, **kwargs)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])),
    batch_size=test_batch_size, shuffle=True, **kwargs)

# Count items in loaded datasets
"""test_data_iter = iter(test_loader)
test_set_size = sum(len(batch[0]) for batch in test_data_iter)
training_data_iter = iter(train_loader)
training_set_size = sum(len(batch[0]) for batch in training_data_iter)

print("Size of:")
print("- Training-set:\t\t{}".format(training_set_size))
print("- Test-set:\t\t{}".format(test_set_size))"""
# TODO add a validation set
# print("- Validation-set:\t{}".format(len(data.validation.labels)))

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


# Plot a sample of test images with their actual class using our helper-function above.
sample_data_iter = iter(train_loader)
images, labels = sample_data_iter.next()
images = images.numpy()[0:9]
labels = labels.numpy()[0:9]
plot_images(images, labels)

conv1_kernel_size = 5
# See https://www.tensorflow.org/versions/r0.12/api_docs/python/nn/convolution
conv1_pad = int((conv1_kernel_size - 1) / 2)
assert ((conv1_kernel_size - 1) % 2 == 0)
conv2_kernel_size = 5
conv2_pad = int((conv2_kernel_size - 1) / 2)


# Pythorch sequential model doesn't include a layer for flattening, oddly enough, so we make our own
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


sequential_model = nn.Sequential(
    nn.Conv2d(in_channels=1, out_channels=16, kernel_size=conv1_kernel_size, padding=conv1_pad),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),
    nn.Conv2d(in_channels=16, out_channels=16, kernel_size=conv2_kernel_size, padding=conv2_pad),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),
    Flatten(),
    nn.Linear(in_features=784, out_features=128),
    nn.ReLU(),
    nn.Linear(in_features=128, out_features=num_classes)# ,
    # nn.CrossEntropyLoss()
)

if cuda:
    sequential_model = sequential_model.cuda()

optimizer = optim.Adam(params=sequential_model.parameters(), lr=1e-3)


print('Training sequential model')

for epoch in range(epochs):
    train(epoch, sequential_model, train_loader)
    test(sequential_model)


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
        # See https://www.tensorflow.org/versions/r0.12/api_docs/python/nn/convolution
        conv1_pad = int((conv1_kernel_size - 1) / 2)
        assert ((conv1_kernel_size - 1) % 2 == 0)
        self.conv1_pad = conv1_pad
        conv2_kernel_size = 5
        conv2_pad = int((conv2_kernel_size - 1) / 2)
        self.conv2_pad = conv2_pad

        self.conv1_w, self.conv1_b = get_conv_weight_and_bias(n_input=1, n_output=16, k1=conv1_kernel_size,
                                                              k2=conv1_kernel_size)
        self.conv2_w, self.conv2_b = get_conv_weight_and_bias(n_input=16, n_output=16, k1=conv2_kernel_size,
                                                              k2=conv2_kernel_size)
        self.fc1_w, self.fc1_b = get_linear_weight_and_bias(n_input=784, n_output=128)
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
        # x = F.log_softmax(input=x, dim=0)
        return x


functional_model = FunctionalModel()
if cuda:
    functional_model.cuda()

optimizer = optim.Adam(params=functional_model.parameters(), lr=1e-3)

print('Training functional model')

for epoch in range(epochs):
    train(epoch, functional_model, train_loader)
    test(functional_model)

# Plot 9 images correctly classified, along with their classification
sample_data_iter = iter(train_loader)
images, labels = sample_data_iter.next()
if cuda:
    images = images.cuda()
images = images[0:9]
labels = labels.numpy()[0:9]
functional_model.eval()  # TODO is it needed?
images = Variable(images, volatile=True)
one_hot_pred = functional_model(images)
predicted = one_hot_pred.data.max(1, keepdim=True)[1]
plot_images(array_from_variable(images), cls_true=labels, cls_pred=array_from_tensor(predicted))

# Plot 9 images that have been misclassified, along with their (incorrect) classification and correct class
test_data_iter = iter(test_loader)
misclassified, prediction, correct_cls = [], [], []
for images, labels in test_data_iter:
    if cuda:
        images = images.cuda()
    images = Variable(images, volatile=True)
    one_hot_pred = functional_model(images)
    predicted = array_from_tensor(one_hot_pred.data.max(1, keepdim=True)[1]).squeeze()
    incorrect = (predicted != labels.numpy())
    misclassified.extend(array_from_variable(images)[incorrect])
    prediction.extend(predicted[incorrect])
    correct_cls.extend(labels.numpy()[incorrect])
    if len(misclassified) >= 9:
        break

plot_images(images=misclassified[0:9],
            cls_true=correct_cls[0:9],
            cls_pred=prediction[0:9])
