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


def save_data_to_file(filename, data):
    """
    Helper function for saving given data to file, each item to a new line.
    """
    with open(os.path.join(data_dir, filename), 'w', encoding='utf8') as file_out:
        print(f'{len(data)} entries in {filename}')
        for entry in data:
            file_out.write(f'{entry}\n')


def process_jsonls(filenames):
    """
    Processes raw jsonls from dataset and creates partitioned files only with
    relevant fields in data directory.

    :filenames: list of filenames (full paths) of jsonls to process
    """
    saved_labels=False
    for filename in filenames:
        input_parse = []
        train_data_x = []
        train_data_y = []
        val_data_x = []
        val_data_y = []
        test_data_x = []
        test_data_y = []
        language = filename[-11:-6]
        
        with open(filename, 'r') as file_input:
            for json_object in file_input:
                input_parse.append(json.loads(json_object))

        for entry in input_parse:
            if entry['partition'] == 'train':
                train_data_x.append(entry['utt'])
                if not saved_labels: train_data_y.append(entry['intent'])
            elif entry['partition'] == 'dev':
                val_data_x.append(entry['utt'])
                if not saved_labels: val_data_y.append(entry['intent'])
            else: # if entry['partition'] == 'test':
                test_data_x.append(entry['utt'])
                if not saved_labels: test_data_y.append(entry['intent'])
            
        save_data_to_file(f'{language}_train_x.json', train_data_x)
        save_data_to_file(f'{language}_val_x.json', val_data_x)
        save_data_to_file(f'{language}_test_x.json', test_data_x)
        if not saved_labels:
            save_data_to_file('train_y.json', train_data_y)
            save_data_to_file('val_y.json', val_data_y)
            save_data_to_file('test_y.json', test_data_y)
            saved_labels = True


if __name__ == '__main__':
    files = get_files()
    print(f'Dataset files found: {files}')
    process_jsonls(files)
