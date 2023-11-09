#!/bin/bash

ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
echo "ROOT: ${ROOT_DIR}"
WORK_DIR=$(dirname "$(dirname "${ROOT_DIR}")")
echo "WORK DIRECTORY: ${WORK_DIR}"

TRAIN_IN="${WORK_DIR}/data/naver-clova-ix-rec/train_img/"
VAL_IN="${WORK_DIR}/data/naver-clova-ix-rec/validation_img/"

TRAIN_OUT="${WORK_DIR}/data/naver-clova-ix-rec/output_train"
VAL_OUT="${WORK_DIR}/data/naver-clova-ix-rec/output_validation"

python ./tool/create_dataset.py --out "${TRAIN_OUT}" --folder "${TRAIN_IN}"
python ./tool/create_dataset.py --out "${VAL_OUT}" --folder "${VAL_IN}"