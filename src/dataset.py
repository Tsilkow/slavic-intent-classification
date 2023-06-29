import json
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase, XLMTokenizer, XLMRobertaTokenizer


class MassiveDatasetBert(Dataset):
    def __init__(
        self,
        json_path: str,
        tokenizer: PreTrainedTokenizerBase,
        labels_values_path='data/labels.json'
    ):
        self._tokenizer = tokenizer
        self._inputs = None
        self._targets = None

        with open(labels_values_path, 'r') as file:
            self.labels_values = json.load(file)
        self.idx_to_label = {i: label for i, label in enumerate(self.labels_values)}
        self.label_to_idx = {label: i for i, label in enumerate(self.labels_values)}

        with open(json_path, 'r') as file:
            data = json.load(file)
        self._encode(data)

    def __len__(self) -> int:
        return len(self._inputs['input_ids'])

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        return self._inputs['input_ids'][index], self._targets[index]

    def _one_hot_encode_labels(self, labels: list[str]) -> torch.Tensor:
        encoded_labels = torch.zeros((len(labels), len(self.labels_values)))
        label_indices = [self.label_to_idx[label] for label in labels]
        encoded_labels[torch.arange(len(labels)), label_indices] = 1.0
        return encoded_labels

    def _decode_one_hot_labels(self, encoded_labels: torch.Tensor) -> list[str]:
        """
        :param encoded_labels: Two dimensional tensor where each row should contain single
        non zero value.
        """
        labels = [self.idx_to_label[torch.argmax(enc_label)] for enc_label in encoded_labels]
        return labels

    def _encode(self, data: dict):
        """
        Encode inputs with tokenizer and outputs into one-hot format.
        """
        self._inputs = self._tokenizer(data['x'], padding='longest', return_tensors='pt')
        self._targets = self._one_hot_encode_labels(data['y'])


class MassiveDatasetT5(Dataset):
    def __init__(
        self,
        json_path: str,
        tokenizer: PreTrainedTokenizerBase
    ):
        self._tokenizer = tokenizer
        self._inputs = None
        self._targets = None

        with open(json_path, 'r') as file:
            data = json.load(file)
        self._encode(data)

    def __len__(self) -> int:
        return len(self._inputs['input_ids'])

    def __getitem__(self, index) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        inputs = {
            'ids': self._inputs['input_ids'][index],
            'attention_mask': self._inputs['attention_mask'][index]
        }
        targets = {
            'ids': self._targets['input_ids'][index],
            'attention_mask': self._targets['attention_mask'][index]
        }
        return inputs, targets

    def _encode(self, data: dict):
        """
        Encode inputs and outputs with tokenizer.
        """
        self._inputs = self._tokenizer(data['x'], padding='longest', return_tensors='pt')
        self._targets = self._tokenizer(data['y'], padding='longest', return_tensors='pt')

class MassiveDatasetXLMR(Dataset):
    def __init__(
        self,
        json_path: str,
        tokenizer: XLMRobertaTokenizer,
        labels_values_path=PATH_PREFIX + "data/labels.json",
    ):
        self._tokenizer = tokenizer
        self._inputs = None
        self._targets = None

        with open(labels_values_path, "r") as file:
            self.labels_values = json.load(file)
        self.idx_to_label = {i: label for i, label in enumerate(self.labels_values)}
        self.label_to_idx = {label: i for i, label in enumerate(self.labels_values)}

        with open(json_path, "r") as file:
            data = json.load(file)
        self._encode(data)

    def __len__(self) -> int:
        return len(self._inputs["input_ids"])

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        item = {
            "input_ids": self._inputs["input_ids"][index],
            "attention_mask": self._inputs["attention_mask"][index],
            "labels": self._targets[index],
        }
        return item

    def _one_hot_encode_labels(self, labels: list[str]) -> torch.Tensor:
        encoded_labels = torch.zeros((len(labels), len(self.labels_values)))
        label_indices = [self.label_to_idx[label] for label in labels]
        encoded_labels[torch.arange(len(labels)), label_indices] = 1.0
        return encoded_labels

    def _decode_one_hot_labels(self, encoded_labels: torch.Tensor) -> list[str]:
        """
        :param encoded_labels: Two dimensional tensor where each row should contain single
        non zero value.
        """
        labels = [
            self.idx_to_label[torch.argmax(enc_label)] for enc_label in encoded_labels
        ]
        return labels

    def _encode(self, data: dict):
        """
        Encode inputs with tokenizer and outputs into one-hot format.
        """
        self._inputs = self._tokenizer(
            data["x"], padding="longest", return_tensors="pt"
        )
        self._targets = self._one_hot_encode_labels(data["y"])


class MassiveDatasetXLMV(Dataset):
    def __init__(
        self,
        json_path: str,
        tokenizer: XLMTokenizer,
        labels_values_path=PATH_PREFIX + "data/labels.json",
    ):
        self._tokenizer = tokenizer
        self._inputs = None
        self._targets = None

        with open(labels_values_path, "r") as file:
            self.labels_values = json.load(file)
        self.idx_to_label = {i: label for i, label in enumerate(self.labels_values)}
        self.label_to_idx = {label: i for i, label in enumerate(self.labels_values)}

        with open(json_path, "r") as file:
            data = json.load(file)
        self._encode(data)

    def __len__(self) -> int:
        return len(self._inputs["input_ids"])

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        item = {
            "input_ids": self._inputs["input_ids"][index],
            "attention_mask": self._inputs["attention_mask"][index],
            "labels": self._targets[index],
        }
        return item

    def _one_hot_encode_labels(self, labels: list[str]) -> torch.Tensor:
        encoded_labels = torch.zeros((len(labels), len(self.labels_values)))
        label_indices = [self.label_to_idx[label] for label in labels]
        encoded_labels[torch.arange(len(labels)), label_indices] = 1.0
        return encoded_labels

    def _decode_one_hot_labels(self, encoded_labels: torch.Tensor) -> list[str]:
        """
        :param encoded_labels: Two dimensional tensor where each row should contain single
        non zero value.
        """
        labels = [
            self.idx_to_label[torch.argmax(enc_label)] for enc_label in encoded_labels
        ]
        return labels

    def _encode(self, data: dict):
        """
        Encode inputs with tokenizer and outputs into one-hot format.
        """
        self._inputs = self._tokenizer(
            data["x"], padding="longest", return_tensors="pt"
        )
        self._targets = self._one_hot_encode_labels(data["y"])
