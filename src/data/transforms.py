from torchvision import transforms as T


def get_train_transforms(img_size=224, test_run=False):
    if test_run:
        img_size = 64  # smaller for fast test runs
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])


def get_test_transforms(img_size=224, test_run=False):
    if test_run:
        img_size = 64  # smaller for fast test runs
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])
