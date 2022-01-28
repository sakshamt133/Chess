from torchvision import transforms

path = "D:\\Datasets\\Computer Vision\\Chess"
batch_size = 32
epochs = 10
lr = 0.0001
in_channels = 3
transforms = transforms.Compose(
    [
        transforms.ToTensor()
    ]
)