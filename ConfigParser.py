import configparser
# configparser load
config = configparser.ConfigParser()
config.read('config.cfg')
if not config.sections():
    raise Exception('config file is missing')

seed = int(config['arg']['seed'])
epochs = int(config['arg']['epochs'])
dataset = config['arg']['dataset']
augmentation = config['arg']['augmentation']
resize = list(map(lambda x: int(x), config['arg']['resize'].split(', ')))
batch_size = int(config['arg']['batch_size'])
valid_batch_size = int(config['arg']['valid_batch_size'])
val_ratio = float(config['arg']['val_ratio'])
model = config['arg']['model']
# choose one of model_names in timm.list_models(pretrained=True)
model_name = config['arg']['model_name']
pretrained = bool(config['arg']['pretrained'])
criterion = config['arg']['criterion']
lr = float(config['arg']['lr'])
optimizer = config['arg']['optimizer']
lr_decay_step = int(config['arg']['lr_decay_step'])
log_interval = int(config['arg']['log_interval'])
name = config['arg']['saving_name']
use_cropped_data = config['arg']['use_cropped_data']
cropped_data_dir = config['arg']['cropped_data_dir']
use_cut_mix = config['arg']['use_cut_mix']
cut_mix_prob = config['arg']['cut_mix_prob']
use_wandb = config['arg']['use_wandb']
wandb_ID = config['arg']['wandb_ID']
wandb_project_name = config['arg']['wandb_project_name']
