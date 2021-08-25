# pstage chronicle

# 파이프라인: 
domain understanding -> data mining -> data analysis -> data processing -> modeling -> training -> deploy

# overview
>COVID-19의 확산으로 우리나라는 물론 전 세계 사람들은 경제적, 생산적인 활동에 많은 제약을 가지게 되었습니다. 우리나라는 COVID-19 확산 방지를 위해 사회적 거리 두기를 단계적으로 시행하는 등의 많은 노력을 하고 있습니다. 과거 높은 사망률을 가진 사스(SARS)나 에볼라(Ebola)와는 달리 COVID-19의 치사율은 오히려 비교적 낮은 편에 속합니다. 그럼에도 불구하고, 이렇게 오랜 기간 동안 우리를 괴롭히고 있는 근본적인 이유는 바로 COVID-19의 강력한 전염력 때문입니다.

>감염자의 입, 호흡기로부터 나오는 비말, 침 등으로 인해 다른 사람에게 쉽게 전파가 될 수 있기 때문에 감염 확산 방지를 위해 무엇보다 중요한 것은 모든 사람이 마스크로 코와 입을 가려서 혹시 모를 감염자로부터의 전파 경로를 원천 차단하는 것입니다. 이를 위해 공공 장소에 있는 사람들은 반드시 마스크를 착용해야 할 필요가 있으며, 무엇 보다도 코와 입을 완전히 가릴 수 있도록 올바르게 착용하는 것이 중요합니다. 하지만 넓은 공공장소에서 모든 사람들의 올바른 마스크 착용 상태를 검사하기 위해서는 추가적인 인적자원이 필요할 것입니다.

>따라서, 우리는 카메라로 비춰진 사람 얼굴 이미지 만으로 이 사람이 마스크를 쓰고 있는지, 쓰지 않았는지, 정확히 쓴 것이 맞는지 자동으로 가려낼 수 있는 시스템이 필요합니다. 이 시스템이 공공장소 입구에 갖춰져 있다면 적은 인적자원으로도 충분히 검사가 가능할 것입니다.

problem restatement: 사람이 카메라 앞에 서면, 정확하게 마스크를 잘 썼는지, 안썼는지, 잘못썼는지 파악해야 한다. 그리고 어떻게? 제대로 안쓰고 있으면 Notify 해야 한다. 평가 방법을 보면... 잘 썼다, 잘못썼다, 안썼다 뿐만 아니라 각 경우에 대해 나이대 역시도 분류 범주에 속해 있다. 각 범주 별로 30대 미만, 30~60대, 60대 초과로 세부 범주가 있다. 그리고 각 범주들 마다 성별로 별개로. 따라서 3 * 2 * 3으로 18개의 범주로 분류해야 한다. 


## tar file in colab
```
!wget -cq https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz
import tarfile
tar = tarfile.open("flower_data.tar.gz")
tar.extractall()
tar.close()
!rm -r flower_data.tar.gz

```

https://colab.research.google.com/github/Shaam93/Building-a-classifer-with-Pytorch/blob/master/Uploading_Data_to_Google_Colab.ipynb#scrollTo=9ze7n0YIpDLR

# data load into notebook

주어진 학습 데이터: 2가지 종류. 1. csv으로 된 info. 각 사람에 대한 정보를 가지고 있따. 2. 사진. 각 사람 마다 7개의 사진. 5개의 잘 쓴 사진. 1개의 안쓴 사진. 1개의 잘못쓴 사진. imbalance data이다. custom loss을 써야 하나? 혹은... sampler을 변경?

나중에 이쪽으로 다시 와야겠다. imbalance 데이터를 어떻게 handling? 관련 논문들 역시 파악해볼 필요가 있다. 

info에 각 파일의 디레토리까지 주어진다. 편하다! 각 디렉토리는 한 사람을 의미. 나이, 인종, 성별로 구분되어 있다. 각 사람의 폴더 안에 incorrect, mask, normal의 사진이 주어진다. stirng + 연산으로 포함해서 각 사진을 가지고 올 수 있을 것이다. 

info에서 각 사람의 분포를 파악해보자. 사람 데이터의 분포는 어떻게 되지? 다른 관점에서도 imbalance가 있을까? 

전체 사람의 수: 2700

남자의 수: 1000, 여자의 수 1500

모든 인종은 asian이다. 


데이터에 있는 사람 나이 분포: 30세 미만 1200, 30~60 1200, 60세 이상 200명이다. 



## 이미지 파일 가져오기
info에 path가 있다. index으로 개별 사람에게 접근 가능하다. 
각 사람 마다 mask, incorrect, normal의 종류가 있음. 

토치의 데이터셋은... 1개 당 1개만 가져와야 한다. 지금은 index 1개 당 7장을 가져오는 형태이다. 따라서 전처리를 해서 파일을 새로 만들어야할까? 

each file - each label. 
label: 

파일을 새로 만들지 말고... img path list만 새로 만들자! 
label은? 
같은 순서로 그.. 만들면 됨.


### 설계

info_path: 에 각 사람 경로 있다.
각 사람에게 mask, incorrect, normal있다.

만들고 싶은 것: 
path[idx] -> mask or incorrect, or normal. picture.
label[idx] -> mask or incorrect, or normal.

0 <= idx <= 19800

전체 경로가 주어져 있음. 
img_dir + person_dir + mark_or_inc_or_nor.

path에서 for 문으로 사람 찾아서 root directory에서 os.path.join으로 연결. 그리고 string list만들어서 status = [mask, incorrect, normal]으로 다시 반복문으로 os join 합침. 최종 directory 완성!

dataset에서 PIL image open으로 이미지 받아온다.

기본 transform은 toTensor만 한다. transform.ToTensor()(img)해서 label이랑 반환.

## labeling
남자이고 마스크 잘 쓰고 어리면 class 0, 
남자, 마스크, 중간 나이 class 1,
여자 안씀 나이 60세 이상 class 17


총 class는 18개이니까... 일단 그냥 조건에 따라 18개의 class을 만드기로 함. 
```
def class_maker
    if male
        if mask
            if age young
                return 0
```
이 함수를 data set에서 작동시켜서 레이블까지 만든다. 

### 내가 마주하는 문제들: 
jpg 뿐만 아니라 png, jpeg 등 확장자가 다양하다. 하나로 통일ㅎ애야 한다. 
나는 원래 try except을 쓰려고 했음. 그런데... jpeg도 있고... 이걸 한번에 전처리하는 것이 더 효율적일까?

무튼 막히는 것이 있다면, 바로 내가 풀려고 하지 말고 일단 검색을 해보자! 더 효율적인 방법이 있을 수 있다.

os.path.exist으로 찾는다. 조건문: if jpg, png, else jpeg.

# 모델링

resnet_18을 불러온다. 에러가 뜸. 모델 받아올 때 tqdm을 사용하는데 여기서 에러 뜸.

```
IProgress not found. Please update jupyter & ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html 
```

valid OK까지 떠도 안됨. 주피터 커널 재부팅 하니까 잘 됨. 해결.

마지막 fc의 output 차원을 우리 데이터 레이블에 맞게 18으로 변경했다. 

fc의 파라미터를 초기화.

```
torch.nn.init.xavier_uniform(model.fc.weight)
model.fc.bias.data.fill_(0.01)
```


## 마주한 문제: 입력 차원이 자꾸 틀림... ㅜ 
dataloader에서 나오면...배치를 신경쓰지 않아야 한다. cnn에서 input channels,output channels, kernel size순서로 받는다. 그래서 들어갈 때 channel 수만 맨 앞으로 넣고 배치는 그냥 생각도 안하면 된다. 채널 수가 뒤에 가 있다면 `permute`, `view`을 사용해서 앞으로 끌어 줌. view는 차원을 명시하는 것이고 permute은 차원의 위ㅣ치는 indice으로 생각해서 바꾸는 것이다. 
원래 이미지도 채널이 앞에  나와서 바로 바꿔서 넣으면 되어씀. 

## 마주한 문제 cross entropy loss 에러
model(batch_in.to(device))으로 하지 말고, to device한 객체를 새로 저장해준다. X = batch_in.to(device), X = batch_out.to(device) 왜냐하면 gpu에 넣은 상태로 다른 gpu 객체들과 loss에 넣고 비교하고 등등 계속 사용하기 때문이다아... 

## f1 score
multiclass일 경우 구현하기에 애매해서 sklearn을 가져와서 썼다. y_true은 레이블이 이미 있다. prediction은... softmax 결과로 확률 값 텐서는 있딷. torch.max(logits, dim = 1)으로 각 batch에서 가장 큰 것 구한다. torch max returns (value,, index). 우리는 predicted class가 필요해서 pred=index을 취해서 y_true와 비교. sklearn의 f1_score(y_true, pred)에 넣는다. 그런데 비교할 떄 list으로 넣어야 한다. 그리고 학습이 끝나고 나서 전체 결과를 각각 list에 넣어서 넣어야 한ㄷ. 
```
y_true = label.cpu().detach().numpy()
for epoch...

    y_pred = []
    ...

    y_pred.append(pred.cpu().detach().numpy())

f1_score(y_true, y_pred)
```

gpu에서 다시 cpu으로 `.cpu().detach().numpy()`

## dataset augmentation with transform

* center crop
* random rotate
* to tensor

* befor augmentation
```
epoch = 3; micro f1: 0.87, macro f1:0.81
```
* after augmentation
```
epoch = 3, micro f1: 86, macro f1:74
```

augmentation을 해서 f1이 감소했다? -> 학습을 더 못했을 것이다. overfitting을 더 많이 감소시킬 수 있다 -> 그렇다면 이 상태에서 에폭을 더 늘려서 학습을 제대로 하면 validation set에서 성능이 더 오를 것이다. 

에폭을 증감시켜 보았다.
* epoch = 3, micro f1: 86, macro f1:74
* epoch = 5. micro f1:91, macro f1:77
* epooch = 10. micro 95, micro 93
* epoch = 20, micro 96, macro 94

가설이 맞았다!

# 이제 imbalance을 해결해보자. 


## PIL image
PIL 패키지의 image을 읽는 모듈. 그냥 진짜 image을 읽어줌. return 하면 이미지가 나온다. 

## validation split in dataset torch
```
dataset, testset = torch.utils.data.random_split(dataset, [18000,900], generator=torch.Generator().manual_seed(42))
```

length가 시작점부터 몇개까지 할것인지를 결정. 

## 마주한 문제
현재... f1와 accuracy가 실제 제출해보면 훨씬 형편없다.

## 문제
loss을 텐서보드에 기ㅣ록할 때 ... 값이 일관적이지 않다?
torch의 loss 이해. 
loss(y_hat, y)가 들어갈 떄 미니배치가 들어간다. 그러면 결과는... 미니배치의 평균 loss가 나온다. 따라서 각 미니배치의 loss을 구할 때 따로 배치 크기로 나눠주지 않아도 된다.
