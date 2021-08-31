import configparser
## configparser load
config = configparser.ConfigParser()
config.read('config.cfg')
if not config.sections():
    raise Exception('config file is missing')


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
data_dir=config['arg']['data_dir']
model_dir=config['arg']['model_dir']