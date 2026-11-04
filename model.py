"""
Fashion-MNIST Classifier in PyTorch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - load_fashion_mnist
def load_fashion_mnist(n_train=10000, n_test=2000):

    base_url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/"

    filenames = [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz"
    ]

    # Download files if they don't already exist
    file_paths = {}

    for filename in filenames:

        file_path = os.path.join(tempfile.gettempdir(), filename)

        if not os.path.exists(file_path):
            urllib.request.urlretrieve(base_url + filename, file_path)

        file_paths[filename] = file_path

    # Read and parse training images
    with gzip.open(file_paths["train-images-idx3-ubyte.gz"], "rb") as f:
        X_train = np.frombuffer(f.read(), dtype=np.uint8, offset=16)
        X_train = X_train.reshape(-1, 28, 28)

    # Read and parse training labels
    with gzip.open(file_paths["train-labels-idx1-ubyte.gz"], "rb") as f:
        y_train = np.frombuffer(f.read(), dtype=np.uint8, offset=8)

    # Read and parse test images
    with gzip.open(file_paths["t10k-images-idx3-ubyte.gz"], "rb") as f:
        X_test = np.frombuffer(f.read(), dtype=np.uint8, offset=16)
        X_test = X_test.reshape(-1, 28, 28)

    # Read and parse test labels
    with gzip.open(file_paths["t10k-labels-idx1-ubyte.gz"], "rb") as f:
        y_test = np.frombuffer(f.read(), dtype=np.uint8, offset=8)

    # Select requested number of samples
    X_train = X_train[:n_train]
    y_train = y_train[:n_train]

    X_test = X_test[:n_test]
    y_test = y_test[:n_test]

    # Convert NumPy arrays to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32) / 255.0
    y_train = torch.tensor(y_train, dtype=torch.int64)

    X_test = torch.tensor(X_test, dtype=torch.float32) / 255.0
    y_test = torch.tensor(y_test, dtype=torch.int64)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test
    }

# Step 2 - FashionDataset
class FashionDataset(Dataset):
    def __init__(self, X, y, mean=0.2860, std=0.3530):
        # TODO: store X, y, mean, std
        super().__init__()
        self.X = X
        self.y = y
        self.mean = mean
        self.std = std

    def __len__(self):
        # TODO
        return len(self.X)

    def __getitem__(self, i):
        # TODO: ((X[i] - mean) / std, y[i])
        return (((self.X[i] - self.mean) / self.std), self.y[i])

# Step 3 - make_loaders (not yet solved)
# TODO: implement

# Step 4 - MLP (not yet solved)
# TODO: implement

# Step 5 - train_one_epoch (not yet solved)
# TODO: implement

# Step 6 - evaluate (not yet solved)
# TODO: implement

# Step 7 - fit (not yet solved)
# TODO: implement

# Step 8 - lr_range_test (not yet solved)
# TODO: implement

# Step 9 - random_search (not yet solved)
# TODO: implement

# Step 10 - test_accuracy (not yet solved)
# TODO: implement

# Step 11 - save_model (not yet solved)
# TODO: implement

# Step 12 - predict_classes (not yet solved)
# TODO: implement

