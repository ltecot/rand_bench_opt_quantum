# Datasets for quantum optimization
# All datasets must be based off the abstract class torch.utils.data.Dataset
# https://pytorch.org/tutorials/beginner/data_loading_tutorial.html#dataset-class

import torch
from torch.utils.data import Dataset

class RandomMixedState(Dataset):
    """Dataset where we're going from the all-0 state to a random mixed state."""

    def __init__(self, purity):
        """
        Arguments:
            purity (float): Purity of the target state
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:]
        landmarks = np.array([landmarks])
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample

# purity = 0.66

# we generate a three-dimensional random vector by sampling
# each entry from a standard normal distribution
v = np.random.normal(0, 1, 3)

# create a random Bloch vector with the specified purity
bloch_v = Variable(
    torch.tensor(np.sqrt(2 * purity - 1) * v / np.sqrt(np.sum(v ** 2))),
    requires_grad=False
)

# array of Pauli matrices (will be useful later)
Paulis = Variable(torch.zeros([3, 2, 2], dtype=torch.complex128), requires_grad=False)
Paulis[0] = torch.tensor([[0, 1], [1, 0]])
Paulis[1] = torch.tensor([[0, -1j], [1j, 0]])
Paulis[2] = torch.tensor([[1, 0], [0, -1]])

# number of qubits in the circuit
nr_qubits = 3
# number of layers in the circuit
nr_layers = 2