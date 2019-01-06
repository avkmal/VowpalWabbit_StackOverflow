
# run python preprocessing code to create text with vowpal wabbit format
python Preprocessing/preprocessing.py stackoverflow.10kk.tsv stackoverflow.vw

# split the data into three parts (train, validation, test)
split -l 1463018 data/stackoverflow.vw data/stackoverflow_

mv data/stackoverflow_aa data/stackoverflow_train.vw
mv data/stackoverflow_ab data/stackoverflow_valid.vw
mv data/stackoverflow_ac data/stackoverflow_test.vw

wc -l data/stackoverflow_*.vw


# get the labels from validation and test sets
cat data/stackoverflow_valid.vw \
    | cut -f 1 -d ' ' > data/stackoverflow_valid_labels.txt
cat data/stackoverflow_test.vw \
    | cut -f 1 -d ' ' > data/stackoverflow_test_labels.txt

echo 'Complete!'