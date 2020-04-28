import openke, torch
from openke.config import Trainer, Tester
from openke.module.model import TransE
from openke.module.loss import MarginLoss, SigmoidLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader
from pathlib import Path

# dataloader for training
train_dataloader = TrainDataLoader(
	in_path = "../openke_data/", 
	nbatches = 512,
	threads = 24, 
	sampling_mode = "normal", 
	bern_flag = 0, 
	filter_flag = 1, 
	neg_ent = 1,
	neg_rel = 0)

# dataloader for test
#test_dataloader = TestDataLoader("../openke_data", "link")


pretrain_init = {'entity': '../concept_glove.max.npy',
 				 'relation': '../relation_glove.max.npy'}
# define the model
transe = TransE(ent_tot=train_dataloader.get_ent_tot(), rel_tot=train_dataloader.get_rel_tot(), dim=100, p_norm=1, margin=1.0,
				norm_flag=True, init='pretrain', init_weights=pretrain_init)


# define the loss function
model = NegativeSampling(model = transe, loss = SigmoidLoss(adv_temperature = 1), batch_size = train_dataloader.get_batch_size())

# train the model
checkpoint_dir = Path('./checkpoint/')
checkpoint_dir.mkdir(exist_ok=True, parents=True)
alpha = 0.001
trainer = Trainer(model=model, data_loader=train_dataloader, train_times=1000, alpha=alpha, use_gpu=torch.cuda.is_available(), opt_method='adam')
trainer.run()

transe.save_checkpoint(f'./checkpoint/transe_{transe.init}.ckpt')

# test the model
#transe.load_checkpoint('./checkpoint/transe.ckpt')
#tester = Tester(model = transe, data_loader = test_dataloader, use_gpu = torch.cuda.is_available())
#tester.run_link_prediction(type_constrain = False)
