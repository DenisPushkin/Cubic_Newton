import torch, torchvision
import torchvision.transforms as transforms


def get_principal_vecs(X, new_dim):
    assert new_dim <= X.shape[1]
    sigma, V = torch.linalg.eigh(X.T @ X)
    princ_vecs = V[:,-new_dim:]
    return princ_vecs


def load_and_transform_mnist(path, new_dim, keep_classes=None):
    mnist_trainset = torchvision.datasets.MNIST(root=path, train=True, download=True, transform=transforms.ToTensor())
    mnist_testset = torchvision.datasets.MNIST(root=path, train=False, download=True, transform=transforms.ToTensor())
    
    if keep_classes is not None:
        train_mask = [c in keep_classes for c in mnist_trainset.targets]
        mnist_trainset.data = mnist_trainset.data[train_mask]
        mnist_trainset.targets = mnist_trainset.targets[train_mask]

        test_mask = [c in keep_classes for c in mnist_testset.targets]
        mnist_testset.data = mnist_testset.data[test_mask]
        mnist_testset.targets = mnist_testset.targets[test_mask]
        
        class_map = {}
        for i, c in enumerate(keep_classes):
            class_map[c] = i
        
        for i, c in enumerate(mnist_trainset.targets):
            mnist_trainset.targets[i] = class_map[c.item()]
        
        for i, c in enumerate(mnist_testset.targets):
            mnist_testset.targets[i] = class_map[c.item()]

    X_train = mnist_trainset.data.reshape(-1,28*28) / 255.
    y_train = mnist_trainset.targets

    X_test = mnist_testset.data.reshape(-1,28*28) / 255.
    y_test = mnist_testset.targets
    
    princ_vecs = get_principal_vecs(X_train, new_dim)
    X_train = X_train @ princ_vecs
    X_test = X_test @ princ_vecs
    
    mnist_dataset = {
        "train_data": X_train,
        "train_targets": y_train,
        "test_data": X_test,
        "test_targets": y_test
    }
    
    return mnist_dataset


def transform_labels_to_binary(labels):
    return torch.tensor(list(map(lambda x: 0 if x<5 else 1, labels)))
