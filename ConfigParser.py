import configparser
# configparser load
config = configparser.ConfigParser()
config.read('config.cfg')
if not config.sections():
    raise Exception('config file is missing')

seed=42 #int(config['arg']['seed'])
epochs=30 #int(config['arg']['epochs'])
dataset='MaskBaseDataset' #config['arg']['dataset']
augmentation='CustomAugmentation' #config['arg']['augmentation']
resize=[260, 200] #list(map(lambda x:int(x), config['arg']['resize'].split(', ')))
batch_size=64 #int(config['arg']['batch_size'])
valid_batch_size=16 #int(config['arg']['valid_batch_size'])
model=config['arg']['model']
optimizer='Adam' #config['arg']['optimizer']
lr=1e-3 #float(config['arg']['lr'])
val_ratio=0.2 #float(config['arg']['val_ratio'])
criterion='label_smoothing' #config['arg']['criterion']
lr_decay_step=100 #int(config['arg']['lr_decay_step'])
log_interval=20 #int(config['arg']['log_interval'])
name=model #config['arg']['name']
model_name = model #config['arg']['model_name'] # choose one of model_names in timm.list_models(pretrained=True)
pretrained = True #bool(config['arg']['pretrained'])
