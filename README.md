# image-classification-level1-14
image-classification-level1-14 created by GitHub Classroom

## config.cfg
* root directory에 config.cfg을 생성한다.
* 아래를 복붙하고 변경한 뒤 train.py으로 실행
```
[arg]
seed = 2141
epochs = 5
dataset = MaskBaseDataset
augmentation = BaseAugmentation
resize = 128, 96
batch_size = 64
valid_batch_size = 1000
model = MyModel
optimizer = Adam
lr = 1e-3
val_ratio = 0.2
criterion = cross_entropy
lr_decay_step = 20
log_interval = 20
name = exp
model_name = efficientnet_b4
pretrained = True
```

## branch info
* T2102: 백재형 baekTree
* T2151: 이나영 

## Google Docs
https://docs.google.com/document/d/1QKYWxBgnkJC-4CD5-rOlD8_7qICw0V8HkjbsrZV0rbg/edit 

# pstage_01_image_classification

## Getting Started    
### Dependencies
- torch==1.6.0
- torchvision==0.7.0                                                              

### Install Requirements
- `pip install -r requirements.txt`

### Training
- `SM_CHANNEL_TRAIN=[train image dir] SM_MODEL_DIR=[model saving dir] python train.py`

### Inference
- `SM_CHANNEL_EVAL=[eval image dir] SM_CHANNEL_MODEL=[model saved dir] SM_OUTPUT_DATA_DIR=[inference output dir] python inference.py`

### Evaluation
- `SM_GROUND_TRUTH_DIR=[GT dir] SM_OUTPUT_DATA_DIR=[inference output dir] python evaluation.py`
