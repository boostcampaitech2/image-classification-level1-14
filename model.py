import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import timm
from efficientnet_pytorch import EfficientNet


class BaseModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = self.avgpool(x)
        x = x.view(-1, 128)
        return self.fc(x)


# Custom Model Template
class MyModel(nn.Module):

        """
        1. 위와 같이 생성자의 parameter 에 num_claases 를 포함해주세요.
        2. 나만의 모델 아키텍쳐를 디자인 해봅니다.
        3. 모델의 output_dimension 은 num_classes 로 설정해주세요.
        """
    def __init__(self, model_name, pretrained=True, num_classes_mask, num_classes_gender, num_classes_age):
        super().__init__()
        self.net = timm.create_model(model_name=model_name, pretrained=True)
        
        '''
timm.list_models(pretrained=True)

['adv_inception_v3',
 'cait_m36_384',
 'cait_m48_448',
 'cait_s24_224',
 'cait_s24_384',
 'cait_s36_384',
 'cait_xs24_384',
 'cait_xxs24_224',
 'cait_xxs24_384',
 'cait_xxs36_224',
 'cait_xxs36_384',
 'coat_lite_mini',
 'coat_lite_small',
 'coat_lite_tiny',
 'coat_mini',
 'coat_tiny',
 'convit_base',
 'convit_small',
 'convit_tiny',
 'cspdarknet53',
 'cspresnet50',
 'cspresnext50',
 'deit_base_distilled_patch16_224',
 'deit_base_distilled_patch16_384',
 'deit_base_patch16_224',
 'deit_base_patch16_384',
 'deit_small_distilled_patch16_224',
 'deit_small_patch16_224',
 'deit_tiny_distilled_patch16_224',
 'deit_tiny_patch16_224',
 'densenet121',
 'densenet161',
 'densenet169',
 'densenet201',
 'densenetblur121d',
 'dla34',
 'dla46_c',
 'dla46x_c',
 'dla60',
 'dla60_res2net',
 'dla60_res2next',
 'dla60x',
 'dla60x_c',
 'dla102',
 'dla102x',
 'dla102x2',
 'dla169',
 'dm_nfnet_f0',
 'dm_nfnet_f1',
 'dm_nfnet_f2',
 'dm_nfnet_f3',
 'dm_nfnet_f4',
 'dm_nfnet_f5',
 'dm_nfnet_f6',
 'dpn68',
 'dpn68b',
 'dpn92',
 'dpn98',
 'dpn107',
 'dpn131',
 'eca_nfnet_l0',
 'eca_nfnet_l1',
 'eca_nfnet_l2',
 'ecaresnet26t',
 'ecaresnet50d',
 'ecaresnet50d_pruned',
 'ecaresnet50t',
 'ecaresnet101d',
 'ecaresnet101d_pruned',
 'ecaresnet269d',
 'ecaresnetlight',
 'efficientnet_b0',
 'efficientnet_b1',
 'efficientnet_b1_pruned',
 'efficientnet_b2',
 'efficientnet_b2_pruned',
 'efficientnet_b3',
 'efficientnet_b3_pruned',
 'efficientnet_b4',
 'efficientnet_el',
 'efficientnet_el_pruned',
 'efficientnet_em',
 'efficientnet_es',
 'efficientnet_es_pruned',
 'efficientnet_lite0',
 'efficientnetv2_rw_m',
 'efficientnetv2_rw_s',
 'ens_adv_inception_resnet_v2',
 'ese_vovnet19b_dw',
 'ese_vovnet39b',
 'fbnetc_100',
 'gernet_l',
 'gernet_m',
 'gernet_s',
 'ghostnet_100',
 'gluon_inception_v3',
 'gluon_resnet18_v1b',
 'gluon_resnet34_v1b',
 'gluon_resnet50_v1b',
 'gluon_resnet50_v1c',
 'gluon_resnet50_v1d',
 'gluon_resnet50_v1s',
 'gluon_resnet101_v1b',
 'gluon_resnet101_v1c',
 'gluon_resnet101_v1d',
 'gluon_resnet101_v1s',
 'gluon_resnet152_v1b',
 'gluon_resnet152_v1c',
 'gluon_resnet152_v1d',
 'gluon_resnet152_v1s',
 'gluon_resnext50_32x4d',
 'gluon_resnext101_32x4d',
 'gluon_resnext101_64x4d',
 'gluon_senet154',
 'gluon_seresnext50_32x4d',
 'gluon_seresnext101_32x4d',
 'gluon_seresnext101_64x4d',
 'gluon_xception65',
 'gmixer_24_224',
 'gmlp_s16_224',
 'hardcorenas_a',
 'hardcorenas_b',
 'hardcorenas_c',
 'hardcorenas_d',
 'hardcorenas_e',
 'hardcorenas_f',
 'hrnet_w18',
 'hrnet_w18_small',
 'hrnet_w18_small_v2',
 'hrnet_w30',
 'hrnet_w32',
 'hrnet_w40',
 'hrnet_w44',
 'hrnet_w48',
 'hrnet_w64',
 'ig_resnext101_32x8d',
 'ig_resnext101_32x16d',
 'ig_resnext101_32x32d',
 'ig_resnext101_32x48d',
 'inception_resnet_v2',
 'inception_v3',
 'inception_v4',
 'legacy_senet154',
 'legacy_seresnet18',
 'legacy_seresnet34',
 'legacy_seresnet50',
 'legacy_seresnet101',
 'legacy_seresnet152',
 'legacy_seresnext26_32x4d',
 'legacy_seresnext50_32x4d',
 'legacy_seresnext101_32x4d',
 'levit_128',
 'levit_128s',
 'levit_192',
 'levit_256',
 'levit_384',
 'mixer_b16_224',
 'mixer_b16_224_in21k',
 'mixer_b16_224_miil',
 'mixer_b16_224_miil_in21k',
 'mixer_l16_224',
 'mixer_l16_224_in21k',
 'mixnet_l',
 'mixnet_m',
 'mixnet_s',
 'mixnet_xl',
 'mnasnet_100',
 'mobilenetv2_100',
 'mobilenetv2_110d',
 'mobilenetv2_120d',
 'mobilenetv2_140',
 'mobilenetv3_large_100',
 'mobilenetv3_large_100_miil',
 'mobilenetv3_large_100_miil_in21k',
 'mobilenetv3_rw',
 'nasnetalarge',
 'nf_regnet_b1',
 'nf_resnet50',
 'nfnet_l0',
 'pit_b_224',
 'pit_b_distilled_224',
 'pit_s_224',
 'pit_s_distilled_224',
 'pit_ti_224',
 'pit_ti_distilled_224',
 'pit_xs_224',
 'pit_xs_distilled_224',
 'pnasnet5large',
 'regnetx_002',
 'regnetx_004',
 'regnetx_006',
 'regnetx_008',
 'regnetx_016',
 'regnetx_032',
 'regnetx_040',
 'regnetx_064',
 'regnetx_080',
 'regnetx_120',
 'regnetx_160',
 'regnetx_320',
 'regnety_002',
 'regnety_004',
 'regnety_006',
 'regnety_008',
 'regnety_016',
 'regnety_032',
 'regnety_040',
 'regnety_064',
 'regnety_080',
 'regnety_120',
 'regnety_160',
 'regnety_320',
 'repvgg_a2',
 'repvgg_b0',
 'repvgg_b1',
 'repvgg_b1g4',
 'repvgg_b2',
 'repvgg_b2g4',
 'repvgg_b3',
 'repvgg_b3g4',
 'res2net50_14w_8s',
 'res2net50_26w_4s',
 'res2net50_26w_6s',
 'res2net50_26w_8s',
 'res2net50_48w_2s',
 'res2net101_26w_4s',
 'res2next50',
 'resmlp_12_224',
 'resmlp_12_distilled_224',
 'resmlp_24_224',
 'resmlp_24_distilled_224',
 'resmlp_36_224',
 'resmlp_36_distilled_224',
 'resmlp_big_24_224',
 'resmlp_big_24_224_in22ft1k',
 'resmlp_big_24_distilled_224',
 'resnest14d',
 'resnest26d',
 'resnest50d',
 'resnest50d_1s4x24d',
 'resnest50d_4s2x40d',
 'resnest101e',
 'resnest200e',
 'resnest269e',
 'resnet18',
 'resnet18d',
 'resnet26',
 'resnet26d',
 'resnet34',
 'resnet34d',
 'resnet50',
 'resnet50d',
 'resnet51q',
 'resnet101d',
 'resnet152d',
 'resnet200d',
 'resnetblur50',
 'resnetrs50',
 'resnetrs101',
 'resnetrs152',
 'resnetrs200',
 'resnetrs270',
 'resnetrs350',
 'resnetrs420',
 'resnetv2_50x1_bit_distilled',
 'resnetv2_50x1_bitm',
 'resnetv2_50x1_bitm_in21k',
 'resnetv2_50x3_bitm',
 'resnetv2_50x3_bitm_in21k',
 'resnetv2_101x1_bitm',
 'resnetv2_101x1_bitm_in21k',
 'resnetv2_101x3_bitm',
 'resnetv2_101x3_bitm_in21k',
 'resnetv2_152x2_bit_teacher',
 'resnetv2_152x2_bit_teacher_384',
 'resnetv2_152x2_bitm',
 'resnetv2_152x2_bitm_in21k',
 'resnetv2_152x4_bitm',
 'resnetv2_152x4_bitm_in21k',
 'resnext50_32x4d',
 'resnext50d_32x4d',
 'resnext101_32x8d',
 'rexnet_100',
 'rexnet_130',
 'rexnet_150',
 'rexnet_200',
 'selecsls42b',
 'selecsls60',
 'selecsls60b',
 'semnasnet_100',
 'seresnet50',
 'seresnet152d',
 'seresnext26d_32x4d',
 'seresnext26t_32x4d',
 'seresnext50_32x4d',
 'skresnet18',
 'skresnet34',
 'skresnext50_32x4d',
 'spnasnet_100',
 'ssl_resnet18',
 'ssl_resnet50',
 'ssl_resnext50_32x4d',
 'ssl_resnext101_32x4d',
 'ssl_resnext101_32x8d',
 'ssl_resnext101_32x16d',
 'swin_base_patch4_window7_224',
 'swin_base_patch4_window7_224_in22k',
 'swin_base_patch4_window12_384',
 'swin_base_patch4_window12_384_in22k',
 'swin_large_patch4_window7_224',
 'swin_large_patch4_window7_224_in22k',
 'swin_large_patch4_window12_384',
 'swin_large_patch4_window12_384_in22k',
 'swin_small_patch4_window7_224',
 'swin_tiny_patch4_window7_224',
 'swsl_resnet18',
 'swsl_resnet50',
 'swsl_resnext50_32x4d',
 'swsl_resnext101_32x4d',
 'swsl_resnext101_32x8d',
 'swsl_resnext101_32x16d',
 'tf_efficientnet_b0',
 'tf_efficientnet_b0_ap',
 'tf_efficientnet_b0_ns',
 'tf_efficientnet_b1',
 'tf_efficientnet_b1_ap',
 'tf_efficientnet_b1_ns',
 'tf_efficientnet_b2',
 'tf_efficientnet_b2_ap',
 'tf_efficientnet_b2_ns',
 'tf_efficientnet_b3',
 'tf_efficientnet_b3_ap',
 'tf_efficientnet_b3_ns',
 'tf_efficientnet_b4',
 'tf_efficientnet_b4_ap',
 'tf_efficientnet_b4_ns',
 'tf_efficientnet_b5',
 'tf_efficientnet_b5_ap',
 'tf_efficientnet_b5_ns',
 'tf_efficientnet_b6',
 'tf_efficientnet_b6_ap',
 'tf_efficientnet_b6_ns',
 'tf_efficientnet_b7',
 'tf_efficientnet_b7_ap',
 'tf_efficientnet_b7_ns',
 'tf_efficientnet_b8',
 'tf_efficientnet_b8_ap',
 'tf_efficientnet_cc_b0_4e',
 'tf_efficientnet_cc_b0_8e',
 'tf_efficientnet_cc_b1_8e',
 'tf_efficientnet_el',
 'tf_efficientnet_em',
 'tf_efficientnet_es',
 'tf_efficientnet_l2_ns',
 'tf_efficientnet_l2_ns_475',
 'tf_efficientnet_lite0',
 'tf_efficientnet_lite1',
 'tf_efficientnet_lite2',
 'tf_efficientnet_lite3',
 'tf_efficientnet_lite4',
 'tf_efficientnetv2_b0',
 'tf_efficientnetv2_b1',
 'tf_efficientnetv2_b2',
 'tf_efficientnetv2_b3',
 'tf_efficientnetv2_l',
 'tf_efficientnetv2_l_in21ft1k',
 'tf_efficientnetv2_l_in21k',
 'tf_efficientnetv2_m',
 'tf_efficientnetv2_m_in21ft1k',
 'tf_efficientnetv2_m_in21k',
 'tf_efficientnetv2_s',
 'tf_efficientnetv2_s_in21ft1k',
 'tf_efficientnetv2_s_in21k',
 'tf_inception_v3',
 'tf_mixnet_l',
 'tf_mixnet_m',
 'tf_mixnet_s',
 'tf_mobilenetv3_large_075',
 'tf_mobilenetv3_large_100',
 'tf_mobilenetv3_large_minimal_100',
 'tf_mobilenetv3_small_075',
 'tf_mobilenetv3_small_100',
 'tf_mobilenetv3_small_minimal_100',
 'tnt_s_patch16_224',
 'tresnet_l',
 'tresnet_l_448',
 'tresnet_m',
 'tresnet_m_448',
 'tresnet_m_miil_in21k',
 'tresnet_xl',
 'tresnet_xl_448',
 'tv_densenet121',
 'tv_resnet34',
 'tv_resnet50',
 'tv_resnet101',
 'tv_resnet152',
 'tv_resnext50_32x4d',
 'twins_pcpvt_base',
 'twins_pcpvt_large',
 'twins_pcpvt_small',
 'twins_svt_base',
 'twins_svt_large',
 'twins_svt_small',
 'vgg11',
 'vgg11_bn',
 'vgg13',
 'vgg13_bn',
 'vgg16',
 'vgg16_bn',
 'vgg19',
 'vgg19_bn',
 'visformer_small',
 'vit_base_patch16_224',
 'vit_base_patch16_224_in21k',
 'vit_base_patch16_224_miil',
 'vit_base_patch16_224_miil_in21k',
 'vit_base_patch16_384',
 'vit_base_patch32_224',
 'vit_base_patch32_224_in21k',
 'vit_base_patch32_384',
 'vit_base_r50_s16_224_in21k',
 'vit_base_r50_s16_384',
 'vit_huge_patch14_224_in21k',
 'vit_large_patch16_224',
 'vit_large_patch16_224_in21k',
 'vit_large_patch16_384',
 'vit_large_patch32_224_in21k',
 'vit_large_patch32_384',
 'vit_large_r50_s32_224',
 'vit_large_r50_s32_224_in21k',
 'vit_large_r50_s32_384',
 'vit_small_patch16_224',
 'vit_small_patch16_224_in21k',
 'vit_small_patch16_384',
 'vit_small_patch32_224',
 'vit_small_patch32_224_in21k',
 'vit_small_patch32_384',
 'vit_small_r26_s32_224',
 'vit_small_r26_s32_224_in21k',
 'vit_small_r26_s32_384',
 'vit_tiny_patch16_224',
 'vit_tiny_patch16_224_in21k',
 'vit_tiny_patch16_384',
 'vit_tiny_r_s16_p8_224',
 'vit_tiny_r_s16_p8_224_in21k',
 'vit_tiny_r_s16_p8_384',
 'wide_resnet50_2',
 'wide_resnet101_2',
 'xception',
 'xception41',
 'xception65',
 'xception71']

'''

        
        
        
        for param in self.net.parameters():
            param.requires_grad = False

        self.linear_mask = nn.Linear(
            in_features=1000, out_features=num_classes_mask, bias=True)
        self.linear_gender = nn.Linear(
            in_features=1000, out_features=num_classes_gender, bias=True)
        self.linear_age = nn.Linear(
            in_features=1000, out_features=num_classes_age, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        x = self.net(x)
        return {'mask': self.linear_mask(x), 'gender': self.linear_gender(x), 'age': self.linear_age(x)}
#         return x

class efficient(nn.Module):
    def __init__(self, num_classes_mask, num_classes_gender, num_classes_age):
        super().__init__()
        self.net = timm.create_model('efficientnet_b3', pretrained=True)

        for param in self.net.parameters():
            param.requires_grad = False

        self.linear_mask = nn.Linear(
            in_features=1000, out_features=num_classes_mask, bias=True)
        self.linear_gender = nn.Linear(
            in_features=1000, out_features=num_classes_gender, bias=True)
        self.linear_age = nn.Linear(
            in_features=1000, out_features=num_classes_age, bias=True)

    def forward(self, x):
        x = self.net(x)
        return {'mask': self.linear_mask(x), 'gender': self.linear_gender(x), 'age': self.linear_age(x)}


# Custom Model Template
class resnet50(nn.Module):
    def __init__(self, num_classes_mask, num_classes_gender, num_classes_age):
        super().__init__()

        self.net = models.resnet50(pretrained=True)

        for param in self.net.parameters():
            param.requires_grad = False   
        
        num_feat = self.net.fc.in_features
        self.net.fc = nn.Sequential()
        self.linear_mask = nn.Linear(
            in_features=num_feat, out_features=num_classes_mask, bias=True)
        self.linear_gender = nn.Linear(
            in_features=num_feat, out_features=num_classes_gender, bias=True)
        self.linear_age = nn.Linear(
            in_features=num_feat, out_features=num_classes_age, bias=True)


    def forward(self, x):
        x = self.net(x)
        return {'mask': self.linear_mask(x), 'gender': self.linear_gender(x), 'age': self.linear_age(x)}
