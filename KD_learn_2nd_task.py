"""
Author: Xiaocong DU
Date: April 2019
Project: Continual learning with mutants

"""

import logging
import os
import sys
import scipy.io as scio
import continualNN
from learning_curve import *
from load_cifar import *
from utils_tool import count_parameters_in_MB
from args import parser
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"]= args.gpu
torch.backends.cudnn.deterministic = True
np.random.seed(args.seed)  # Python random module.
random.seed(args.seed)
torch.manual_seed(args.seed)
log_path = './log_KD_learn_2nd.txt'.format()

# if os.path.exists(os.path.join('../results/',log_path)):
# 	os.remove(os.path.join('../results/',log_path))

log_format = '%(asctime)s   %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
	format=log_format, datefmt='%m/%d %I:%M%p')
fh = logging.FileHandler(os.path.join('../results/',log_path))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
logging.info("args = %s", args)

logging.info("\n\n************************KD_learning_2nd_task.py Experiment starts here************************")
logging.info("=============================== Get previous mask file ===================================")
save_mask_file = '../mask_library/mask_{}_seed_{}.pickle'.format(args.join_mask_rate, args.seed)
save_mask_fileR = '../mask_library/mask_{}_seed_{}_reverse.pickle'.format(args.join_mask_rate, args.seed)


secondTask = continualNN.ContinualNN()


task_list = secondTask.create_task()



#
logging.info("========================================== Get KD targets ==============================================")
model_NA = secondTask.initial_network(task_id =0)
secondTask.initialization(args.lr_mutant, args.lr_mutant_step_size,  args.weight_decay_2)
# secondTask.load_mutant(0, 0, model_NA)
task_id = 1
current_task = task_list[task_id]
partial_trainsetLoader, partial_testsetLoader = get_dataset_cifar(current_task, -1* args.classes_per_task)
for batch_idx, (data, target) in enumerate(partial_trainsetLoader):
	print('batch {} Current task:{}'.format(batch_idx, target[0:10]))
	break


current_task = task_list[0]
partial_trainsetLoader_0, partial_testsetLoader_0 = get_dataset_cifar(current_task, 0 * args.classes_per_task)

current_task = task_list[1]
partial_trainsetLoader_1, partial_testsetLoader_1 = get_dataset_cifar(current_task, 0 * args.classes_per_task)

KD_target_list = secondTask.obtain_KD_target(partial_trainsetLoader_mix)
logging.info('KD_target obtained, length: {} shape of KD_target[0] {}\n'.format(len(KD_target_list), np.shape(KD_target_list[0])))

print('KD_target example',KD_target_list[0][0, :])

previous_task = task_list[0]
current_task = task_list[1]
partial_trainsetLoader_mix, partial_testsetLoader_mix = get_partial_dataset_cifar(0, [previous_task, current_task], num_images = [3000, 25000]) # 2000, 25000for cifar10
for batch_idx, (data, target) in enumerate(partial_trainsetLoader_mix):
	print('batch {} Current task: {}'.format(batch_idx, target[0:10]))
	break

logging.info("=============================== train current task with KD target ===================================")

model_2ndTask = secondTask.initial_network(task_id =1)
secondTask.initialization(args.lr, args.lr_step_size,  args.weight_decay_2)

#
partial_trainsetLoader, partial_testsetLoader = get_dataset_cifar(current_task, 0* args.classes_per_task)
for batch_idx, (data, target) in enumerate(partial_trainsetLoader):
	print('batch {} Current task: {}'.format(batch_idx, target[0:10]))
	break

secondTask.load_mutant_specify_FC(0, 0)
# secondTask.save_mutant(66, 1)
logging.info('>>>>>>>>>>>>>>>>>>>>>>>>>> This training on previous T0 testing accu is : %.4f', secondTask.test(partial_testsetLoader_0))

train_acc = np.zeros([1, args.num_epoch])
test_acc = np.zeros([1, args.num_epoch])


for epoch in range(0, args.num_epoch):
	train_acc[0, epoch] = secondTask.train_with_mask_with_KD(epoch, partial_trainsetLoader_mix, save_mask_file, save_mask_fileR, KD_target_list, len_onehot= len(current_task)*2)
	test_acc[0, epoch] = secondTask.test(partial_testsetLoader_mix)
secondTask.save_mutant(77, 1)
file_name = '../results/KD_learn_2nd_task_acc{0:.4f}_NBC0{1}_mask{2}_joinM{3}'.format(test_acc[0, -1], args.NB_C0, args.mask_rate, args.join_mask_rate)
scio.savemat(file_name+'.mat', {'train_acc':train_acc, 'test_acc':test_acc})
plot_curve(file_name, '../results', Multi_running=False)


