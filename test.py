import timm
from pprint import pprint
model_names = timm.list_models('*vit*')
pprint(model_names)