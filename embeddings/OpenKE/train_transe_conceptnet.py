import openke
from openke.config import Trainer, Tester
from openke.module.model import TransE
from openke.module.loss import MarginLoss, SigmoidLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader
import torch

# dataloader for training
train_dataloader = TrainDataLoader(
	in_path = "../openke_data/", 
	nbatches = 512,
	threads = 8, 
	sampling_mode = "normal", 
	bern_flag = 0, 
	filter_flag = 1, 
	neg_ent = 1,
	neg_rel = 0)

# dataloader for test
#test_dataloader = TestDataLoader("../openke_data", "link")

# define the model
transe = TransE(
	ent_tot = train_dataloader.get_ent_tot(),
	rel_tot = train_dataloader.get_rel_tot(),
	dim = 100, 
	p_norm = 1, 
    margin=1.0,
	norm_flag = True)


# define the loss function
model = NegativeSampling(model = transe, loss = SigmoidLoss(adv_temperature = 1), batch_size = train_dataloader.get_batch_size())

# train the model
<<<<<<< HEAD
trainer = Trainer(model = model, data_loader = train_dataloader, train_times = 1000, alpha = 1.0, use_gpu = True, opt_method='adam')
=======
trainer = Trainer(model = model, data_loader = train_dataloader, train_times = 1000, alpha = 1.0, use_gpu = torch.cuda.is_available(), opt_method='adam')
>>>>>>> c92ed30acd1521fd3057dc659e6b0b10785258ac
trainer.run()
transe.save_checkpoint('./checkpoint/transe.ckpt')

# test the model
#transe.load_checkpoint('./checkpoint/transe.ckpt')
<<<<<<< HEAD
#tester = Tester(model = transe, data_loader = test_dataloader, use_gpu = True)
=======
#tester = Tester(model = transe, data_loader = test_dataloader, use_gpu = torch.cuda.is_available())
>>>>>>> c92ed30acd1521fd3057dc659e6b0b10785258ac
#tester.run_link_prediction(type_constrain = False)
