SRC=visda_3d
TGT=visda_2d
SRC_GRAY2RGB=0
TGT_GRAY2RGB=0
SRC_DIR=data/$SRC
TGT_DIR=data/$TGT
SRC_MODEL=./model/pretrained/$SRC
TGT_MODEL=./model/${SRC}_to_${TGT}/target
ADV_MODEL=./model/${SRC}_to_${TGT}/adversary
MODEL=resnet_v1_101
LR_CLS=0.0001
LR_ADA=0.0002
WD=0.00002
ADV_LEAKY=True
NUM_ITERS_CLS=10000
NUM_ITERS_ADA=20000
IMAGE_SIZE=224
SPLIT=train
SPLIT_TEST=test
BATCH_SIZE_CLS=32
BATCH_SIZE_ADA=16
SOLVER=adam

TRAIN_CLASSIFIER=1
TRAIN_ADDA=1
TEST=0
CLEAN=0

if [ ! -d $SRC_MODEL ]; then
    mkdir -p $SRC_MODEL
fi

if [ ! -d $TGT_MODEL ]; then
    mkdir -p $TGT_MODEL
fi

if [ ! -d $ADV_MODEL ]; then
    mkdir -p $ADV_MODEL
fi

if [[ $CLEAN -eq 1 ]]; then
    echo rm -rf $SRC_MODEL/* $TGT_MODEL/* $ADV_MODEL/*
    rm -rf $SRC_MODEL/* $TGT_MODEL/* $ADV_MODEL/*
fi

if [[ $TRAIN_CLASSIFIER -eq 1 ]]; then
    echo python train_classifier.py \
                    --solver $SOLVER \
                    --learning_rate $LR_CLS \
                    --weight_decay $WD \
                    --model_path $SRC_MODEL \
                    --num_iters $NUM_ITERS_CLS \
                    --gray2rgb $SRC_GRAY2RGB \
                    --dataset $SRC \
                    --split $SPLIT \
                    --dataset_dir $SRC_DIR \
                    --image_size $IMAGE_SIZE \
                    --model $MODEL \
                    --batch_size $BATCH_SIZE_CLS

    python train_classifier.py \
                    --solver $SOLVER \
                    --learning_rate $LR_CLS \
                    --weight_decay $WD \
                    --model_path $SRC_MODEL \
                    --num_iters $NUM_ITERS_CLS \
                    --gray2rgb $SRC_GRAY2RGB \
                    --dataset $SRC \
                    --split $SPLIT \
                    --dataset_dir $SRC_DIR \
                    --image_size $IMAGE_SIZE \
                    --model $MODEL \
                    --batch_size $BATCH_SIZE_CLS
fi

if [[ $TRAIN_ADDA -eq 1 ]]; then
    echo python train_adda.py \
                    --model $MODEL \
                    --source_dataset $SRC \
                    --source_dataset_dir $SRC_DIR \
                    --source_gray2rgb $SRC_GRAY2RGB \
                    --source_model_path $SRC_MODEL \
                    --target_dataset $TGT \
                    --target_dataset_dir $TGT_DIR \
                    --target_gray2rgb $TGT_GRAY2RGB \
                    --target_model_path $TGT_MODEL \
                    --split $SPLIT \
                    --adversary_model_path $ADV_MODEL \
                    --learning_rate $LR_CLS \
                    --num_iters $NUM_ITERS_ADA \
                    --image_size $IMAGE_SIZE \
                    --adversary_leaky $ADV_LEAKY \
                    --batch_size $BATCH_SIZE_ADA

    python train_adda.py \
                    --model $MODEL \
                    --source_dataset $SRC \
                    --source_dataset_dir $SRC_DIR \
                    --source_gray2rgb $SRC_GRAY2RGB \
                    --source_model_path $SRC_MODEL \
                    --target_dataset $TGT \
                    --target_dataset_dir $TGT_DIR \
                    --target_gray2rgb $TGT_GRAY2RGB \
                    --target_model_path $TGT_MODEL \
                    --split $SPLIT \
                    --adversary_model_path $ADV_MODEL \
                    --learning_rate $LR_ADA \
                    --num_iters $NUM_ITERS_ADA \
                    --image_size $IMAGE_SIZE \
                    --adversary_leaky $ADV_LEAKY \
                    --batch_size $BATCH_SIZE_ADA
fi

if [[ $TEST -eq 1 ]]; then
    echo python test_classifier.py \
                    --model_path $TGT_MODEL \
                    --gray2rgb $TGT_GRAY2RGB \
                    --dataset $TGT_DATASET \
                    --split $SPLIT_TEST \
                    --dataset_dir $TGT_DIR \
                    --image_size $IMAGE_SIZE \
                    --model $MODEL  \
                    --batch_size $BATCH_SIZE_CLS

    python test_classifier.py \
                    --model_path $TGT_MODEL \
                    --gray2rgb $TGT_GRAY2RGB \
                    --dataset $TGT \
                    --split $SPLIT_TEST \
                    --dataset_dir $TGT_DIR \
                    --image_size $IMAGE_SIZE \
                    --model $MODEL  \
                    --batch_size $BATCH_SIZE_CLS
fi
