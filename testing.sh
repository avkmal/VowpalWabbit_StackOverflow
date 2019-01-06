# merge training and validation sets as new training set
cp data/stackoverflow_train.vw data/stackoverflow_train_valid.vw
cat data/stackoverflow_valid.vw >> data/stackoverflow_train_valid.vw

python Final_test/final_testing.py