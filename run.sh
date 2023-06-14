#!/usr/bin/env bash


dataset_dir='dataset/'
dataset_url='https://amazon-massive-nlu-dataset.s3.amazonaws.com/'
dataset_filename='amazon-massive-dataset-1.0.tar.gz'
dataset_version='1.0'


download_dataset () {
    if ! [ -f $dataset_dir/$dataset_filename ]
    then
	curl $dataset_url$dataset_filename -o $dataset_dir$dataset_filename
    fi
    cd $dataset_dir
    
    for language in "$@"
    do
	tar -zxvf $dataset_filename "1.0/data/"$language".jsonl" --strip-components 2
    done
    cd ..
}


if ! [ -f $data_dir/'train.json' ] || ! [ -f $data_dir/'val.json' ] || ! [ -f $data_dir/'test.json' ]
then
    download_dataset pl-PL
    python organize_data.py
fi

