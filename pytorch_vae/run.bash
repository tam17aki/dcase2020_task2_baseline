#!/bin/bash

date_str=$(date +%Y%m%d_%H-%M-%S)

python3 00_train.py -d \
        >& log/train_${date_str}.log

# python3 01_test.py -d \
#         >& log/test_${date_str}.log

# after launch
# python3 00_train.py -e \
#         >& log/train_eval_$(date +%Y%m%d_%H-%M-%S).log
