import os
import simplejson as json
import numpy as np


pad_token = '#'
dataset_dir = 'dataset/'
data_dir = 'data/'


def get_files():
    """
    Returns list of filenames (full paths) from dataset directory with extension '.jsonl'.
    """
    result = []
    entries = os.scandir(dataset_dir)
    for entry in entries:
        if entry.is_file and entry.name[-6:] == '.jsonl':
            result.append(entry.path)
    return result


def save_data_to_file(filename, data, flat=False):
    """
    Helper function for saving given data to file, each item to a new line.
    """
    with open(os.path.join(data_dir, filename), 'w', encoding='utf8') as file_out:
        entries_total = len(data) if flat else len(data['x'])
        print(f'{entries_total} entries in {filename}')
        json.dump(data, file_out, ensure_ascii=False)


def process_jsonls(filenames):
    """
    Processes raw jsonls from dataset and creates partitioned files only with
    relevant fields in data directory.

    :filenames: list of filenames (full paths) of jsonls to process
    """
    for filename in filenames:
        input_parse = []
        train_data = {'x': [], 'y': []}
        val_data = {'x': [], 'y': []}
        test_data = {'x': [], 'y': []}
        language = filename[-11:-6]
        
        with open(filename, 'r') as file_input:
            for json_object in file_input:
                input_parse.append(json.loads(json_object))

        for entry in input_parse:
            if entry['partition'] == 'train':
                train_data['x'].append(entry['utt'])
                train_data['y'].append(entry['intent'])
            elif entry['partition'] == 'dev':
                val_data['x'].append(entry['utt'])
                val_data['y'].append(entry['intent'])
            else: # if entry['partition'] == 'test':
                test_data['x'].append(entry['utt'])
                test_data['y'].append(entry['intent'])

        tmp = os.path.join(data_dir, language)
        if not os.path.exists(tmp): os.makedirs(tmp)
        save_data_to_file(f'{language}/train.json', train_data)
        save_data_to_file(f'{language}/val.json', val_data)
        save_data_to_file(f'{language}/test.json', test_data)

    unique_labels = list(set(train_data['y']))
    save_data_to_file('labels.json', unique_labels, True)


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
    
    return padded_tensor, tensor_lengths


if __name__ == '__main__':
    files = get_files()
    print(f'Dataset files found: {files}')
    process_jsonls(files)
