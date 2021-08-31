import configparser
# configparser load
config = configparser.ConfigParser()
config.read('config.cfg')
if not config.sections():
    raise Exception('config file is missing')


<<<<<<< HEAD
seed = int(config['arg']['seed'])
epochs = int(config['arg']['epochs'])
dataset = config['arg']['dataset']
augmentation = config['arg']['augmentation']
resize = list(map(lambda x: int(x), config['arg']['resize'].split(', ')))
batch_size = int(config['arg']['batch_size'])
valid_batch_size = int(config['arg']['valid_batch_size'])
model = config['arg']['model']
optimizer = config['arg']['optimizer']
lr = float(config['arg']['lr'])
val_ratio = float(config['arg']['val_ratio'])
criterion = config['arg']['criterion']
lr_decay_step = int(config['arg']['lr_decay_step'])
log_interval = int(config['arg']['log_interval'])
name = config['arg']['name']
data_dir = config['arg']['data_dir']
model_dir = config['arg']['model_dir']
=======
seed=int(config['arg']['seed'])
epochs=int(config['arg']['epochs'])
dataset=config['arg']['dataset']
augmentation=config['arg']['augmentation']
resize=list(map(lambda x:int(x), config['arg']['resize'].split(', ')))
batch_size=int(config['arg']['batch_size'])
valid_batch_size=int(config['arg']['valid_batch_size'])
model=config['arg']['model']
optimizer=config['arg']['optimizer']
lr=float(config['arg']['lr'])
val_ratio=float(config['arg']['val_ratio'])
criterion=config['arg']['criterion']
lr_decay_step=int(config['arg']['lr_decay_step'])
log_interval=int(config['arg']['log_interval'])
name=config['arg']['name']
model_name = config['arg']['model_name'] # choose one of model_names in timm.list_models(pretrained=True)
pretrained = bool(config['arg']['pretrained'])
>>>>>>> d04574a535678b05f7948e45b38b8468c784db06
