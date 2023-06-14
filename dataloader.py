import numpy as np
import torch


class PaddingDataloader




def pad_tensor(tensor):
    """
    Returns padded tensor up to the longest utterance in batch.

    :tensor: batch of tensors to be padded
    """
    tensor_lengths = [len(utterance) for utterance in tensor]
    longest_sent = max(tensor_lengths)
    batch_size = len(tensor)
    padded_tensor = np.ones((batch_size, longest_sent)) * pad_token

    for i, x_len in enumerate(tensor_lengths):
        utterance = tensor[i]
        padded_tensor[i, 0:x_len] = utterance[:x_len]

    padded_tensor = torch.LongTensor(padded_tensor)
    return padded_tensor, tensor_lengths


def load_into_tensors(data_parse, tokenizer, one_hot_encoder, indices):
    """
    Loads data from json parse into tensors.

    :data_parse: json object with x and y lists
    :tokenizer: tokenizer for given textual input
    :one_hot_encoder: object with learned output to one-hot encoding labels
    :indices: list of indices of data_parse lists to be put in tensor
    """
    output_values_total = len(set(data_parse['x']))
    x_tensors = []
    y_tensors = []
    for i in indices:
        x_tokenized = tokenizer(data_parse['x'][i])
        x_tensors.append(pad_tensor(x_tokenized))
        y_tensors.append(one_hot_encoder(data_parse['y'][i], output_values_total))

    return torch.stack(x_tensors), torch.stack(y_tensors)


def create_batch()


def create_dataloader(partition):
    '''
    Creates pytorch dataloader
    
    :partition: poriton of data to be put in dataloader; has to be 'train', 'val' or 'test'
    '''
    assert partition in ['train', 'val', 'test']
    pass
