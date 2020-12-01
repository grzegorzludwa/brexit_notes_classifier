# Brexit Notes Classifier

This is simple Brexit Notes Classifier. Every press note is scored as pro (1), neutral (0) or against (-1) Brexit. Uses Complement Naive Bayes or LinearSVC classifier.

<br>

## 0. Prerequisites:

- python (3.6)
- scikit-learn (0.23.0)
- nltk (3.6)

#### Example installation:

```
sudo apt install python3.6
python3 -m pip install nltk==3.6
python3 -m pip install scikit-learn==0.23.0
```
<br>

## 1. Data file format.
>Repository contains example [database.txt](database.txt)

**Score**"\t"**Press note**"\n"
```
1"\t"I love Brexit"\n"
0"\t"I don't care about it"\n"
-1"\t"I hate Brexit"\n"
```

<br>

## 2. Script for preparing train and test data.
>CAUTION! Use only for small files (max 10MB)

```
usage: split_data.py [-h] [--test_size TEST_SIZE] filename

Split file into test and train files. Names of created files are:
test_{filename} and train_{filename} CAUTION! Use only for small files (max 10MB)

positional arguments:
  filename              File containing data

optional arguments:
  -h, --help            show this help message and exit
  --test_size TEST_SIZE
                        Part of file chosen as test data. [0-1]
```                        
<br>

## 3. Main script usage.

```
usage: classify.py [-h] [-t TRAIN_FILE] [-p] [-c {nb,svc}] [-s] [-n N_OF_BEST]
                   [--csv DELIMETER]
                   TEST_FILE

Classify comments sentiments about Brexit. All comments should be categorized
as positive/neutral/negative (1/0/-1)

positional arguments:
  TEST_FILE             Filename with test data

optional arguments:
  -h, --help            show this help message and exit
  -t TRAIN_FILE, --train TRAIN_FILE
                        Name of file with training data
  -p, --print_classes   Print real and predicted classes of test file
  -c {nb,svc}, --classifier {nb,svc}
                        Chsose classifier during training. nb - Complement
                        Naive Bayes, svc - LinearSVC
  -s, --save            Overwrite current classifier
  -n N_OF_BEST, --n_of_best N_OF_BEST
                        Number of best feautures
  --csv DELIMETER       Input data is a csv type. DELIMETER - char between
                        data columns
```
<br>

## 4. Example workflow:

#### Split database into test and train set:
```
$ python3 split_data.py --test_size 0.3 database.txt
```

#### Train LinearSVC classifier using 100 best features:
```
$ python3 classify.py -t train_database.txt test_database.txt -n 100 -c svc
```

#### Print predicted classes:
```
$ python3 classify.py test_database.txt -p
```
