import os
import simplejson as json


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


def process_jsonls(filenames):
    """
    Processes raw jsonls from dataset and creates partitioned files only with
    relevant fields in data directory.

    :filenames: list of filenames (full paths) of jsonls to process
    """
    train_data = {'x': [], 'y': []}
    val_data = {'x': [], 'y': []}
    test_data = {'x': [], 'y': []}
    
    for filename in filenames:
        input_parse = []
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

    with open(os.path.join(data_dir, 'train.json'), 'w', encoding='utf8') as file_out:
        print(str(len(train_data['y']))+' entries in train.json')
        json.dump(train_data, file_out, ensure_ascii=False)
        
    with open(os.path.join(data_dir, 'val.json'), 'w', encoding='utf8') as file_out:
        print(str(len(val_data['y']))+' entries in val.json')
        json.dump(val_data, file_out, ensure_ascii=False)
        
    with open(os.path.join(data_dir, 'test.json'), 'w', encoding='utf8') as file_out:
        print(str(len(test_data['y']))+' entries in test.json')
        json.dump(test_data, file_out, ensure_ascii=False)


if __name__ == '__main__':
    files = get_files()
    print(f'Dataset files found: {files}')
    process_jsonls(files)
