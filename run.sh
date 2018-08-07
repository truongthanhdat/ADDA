#Training Configuration
train_src=1
train_tgt=0
train_adda=1
test_src=0
test_tgt=0
test_adda=0
clean_log=1
clean_dir=0

#Model Config
output=output
data=data
src=visda_3d
tgt=visda_2d
model=resnet_v1_50

src_data_dir=$data/$src
tgt_data_dir=$data/$tgt

src_model_dir=$output/$model/pretrained_${model}_${src}_only
tgt_model_dir=$output/$model/pretrained_${model}_${tgt}_only

adda_model_dir=$ouput/$model/adda_${model}_${src}_to_${tgt}/target
adve_model_dir=$ouput/$model/adda_${model}_${src}_to_${tgt}/discriminator

src_gray2rgb=0
tgt_gray2rgb=0
image_size=224
split_train=train
split_test=test
feat_name=global_pool

#Learning Cofig
num_iters_cls=7000
num_iters_adda=10000

solver_cls=adam
solver_adda=adam

lr_cls=0.0001
lr_adda_gen=0.000001
lr_adda_dis=0.00001

weight_decay=0.00002
adv_leaky=True

batch_size_cls=64
batch_size_adda=32

#Traning Source Only
if [[ $train_src -eq 1 ]]; then
    if [ ! -d $src_model_dir ]; then #Create source model path
        mkdir -p $src_model_dir
    fi

    if [[ $clean_log -eq 1 ]]; then #Delete Log
        rm -rf $src_model_dir/events.*
    fi

    if [[ $clean_dir -eq 1 ]]; then
        rm -rf $src_model_dir/*
    fi

    python train_classifier.py \
        --solver $solver_cls \
        --learning_rate $lr_cls \
        --weight_decay $weight_decay \
        --num_iters $num_iters_cls \
        --model $model \
        --model_path $src_model_dir \
        --dataset $src \
        --split $split_train \
        --dataset_dir $src_data_dir \
        --gray2rgb $src_gray2rgb \
        --batch_size $batch_size_cls \
        --image_size $image_size
fi

#Traning Target Only
if [[ $train_tgt -eq 1 ]]; then
    if [ ! -d $tgt_model_dir]; then #Create target model path
        mkdir -p $tgt_model
    fi

    if [[ $clean_log -eq 1 ]]; then #Delete Log
        rm -rf $tgt_model_dir/events.*
    fi

    if [[ $clean_dir -eq 1 ]]; then
        rm -rf $tgt_model_dir/*
    fi

    python train_classifier.py \
        --solver $solver_cls \
        --learning_rate $lr_cls \
        --weight_decay $weight_decay \
        --num_iters $num_iters_cls \
        --model $model \
        --model_path $tgt_model_dir \
        --dataset $tgt \
        --split $split_train \
        --dataset_dir $tgt_data_dir \
        --gray2rgb $tgt_gray2rgb \
        --batch_size $batch_size_cls \
        --image_size $image_size
fi

#Traning Adversarial Discriminative Domain Adaptaion
if [[ $train_adda -eq 1 ]]; then
    if [ ! -d $adda_model_dir ]; then #Create adda model path
        mkdir -p $adda_model_dir
    fi

    if [ ! -d $adve_model_dir ]; then #Create adve model path
        mkdir -p $adve_model_dir
    fi

    if [[ $clean_log -eq 1 ]]; then #Delete Log
        rm -rf $adda_model_dir/events.*
        rm -rf $adve_model_dir/events.*
    fi

    if [[ $clean_dir -eq 1 ]]; then
        rm -rf $adda_model_dir/*
        rm -rf $adve_model_dir/*
    fi

    python train_adda.py \
        --solver $solver_adda \
        --learning_rate_generator $lr_adda_gen \
        --learning_rate_discriminator $lr_adda_dis \
        --num_iters $num_iters_adda \
        --model $model \
        --source_model_path $src_model_dir \
        --target_model_path $adda_model_dir \
        --adversary_model_path $adve_model_dir \
        --source_dataset $src \
        --source_dataset_dir $src_data_dir \
        --src_gray2rgb $src_gray2rgb \
        --target_dataset $tgt \
        --target_dataset_dir $tgt_data_dir \
        --tgt_gray2rgb $tgt_gray2rgb \
        --split $split_train \
        --adversary_leaky $adv_leaky \
        --batch_size $batch_size_adda \
        --image_size $image_size \
        --feature_name $feat_name
fi

#Test Source Only
if [[ $test_src -eq 1 ]]; then
    python test_classifier.py \
        --model $model \
        --model_path $src_model_dir \
        --dataset $src \
        --dataset_dir $src_data_dir \
        --split $split_test \
        --gray2rgb $src_gray2rgb \
        --batch_size $batch_size_cls \
        --image_size $image_size
fi

#Test Target Only
if [[ $test_tgt -eq 1 ]]; then
    python test_classifier.py \
        --model $model \
        --model_path $tgt_model_dir \
        --dataset $tgt\
        --dataset_dir $tgt_data_dir \
        --split $split_test \
        --gray2rgb $tgt_gray2rgb \
        --batch_size $batch_size_cls \
        --image_size $image_size
fi

#Test ADDA
if [[ $test_adda -eq 1 ]]; then
    python test_classifier.py \
        --model $model \
        --model_path $adda_model_dir \
        --dataset $tgt\
        --dataset_dir $tgt_data_dir \
        --split $split_test \
        --gray2rgb $tgt_gray2rgb \
        --batch_size $batch_size_cls \
        --image_size $image_size
fi
