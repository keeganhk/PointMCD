import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
prj_root = os.path.abspath(os.path.dirname(os.getcwd()))
sys.path.append(prj_root)
from cdbs.tools import *
from cdbs.miscs import *
from cdbs.modules import *
from cdbs.datasets import *
from cdbs.distillers import *
data_root = '../data'
ckpt_root = '../ckpt'
expt_root = '../expt'
dataset_folder = os.path.join(data_root, 'ShapeNetCore')
ckpt_params = os.path.join(ckpt_root, 'teacher', 'CNN_Backbone_Wrapper_Pretrained_ShapeNetCore.pth')


# training
net = CNN_Backbone_Pretrainer().cuda()
train_bs = 55
max_lr = 1e-4
min_lr = 5e-6
tr_itr = 100000
show_every = 500
optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=max_lr, weight_decay=1e-5)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=tr_itr, eta_min=min_lr)
criterion = nn.L1Loss()
net.train()
for it in tqdm(range(1, tr_itr+1)):
    num_samples = 0
    average_loss = 0
    vi, _, _ = ShapeNetCore_Balanced_SVI_Loader(os.path.join(data_root, 'ShapeNetCore'), train_bs)
    vi = vi.cuda()
    batch_size = vi.size(0)
    optimizer.zero_grad()      
    rec_vi = net(vi)
    loss = criterion(rec_vi, vi)
    loss.backward()
    optimizer.step()
    scheduler.step()
    num_samples += batch_size
    average_loss += loss.item() * batch_size
    if np.mod(it, show_every) == 0:
        print('iteration: {}, train loss: {}'.format(align_number(it, 6), np.around(average_loss/num_samples, 6)))
        torch.save(net.state_dict(), ckpt_params)
        num_samples = 0
        average_loss = 0


# exporting
pretrainer = CNN_Backbone_Pretrainer().cuda()
pretrainer.load_state_dict(torch.load(ckpt_params))
pretrainer.eval()
net = pretrainer.cnn_backbone
net.eval()
class_list = [line.strip() for line in open(os.path.join(dataset_folder, 'class_list.txt'), 'r')]
train_model_list = [line.strip() for line in open(os.path.join(dataset_folder, 'total_list.txt'), 'r')]
load_folder = os.path.join(dataset_folder, 'mv_image_16')
expt_folder = os.path.join(expt_root, 'teacher_uns', 'ShapeNetCore')
for model_name in tqdm(train_model_list):
    class_name = model_name[:-5]    
    suffixes = ['1_1', '1_2', '1_3', '1_4', '1_5', '1_6', '1_7', '1_8', '2_1', '2_2', '2_3', '2_4', '2_5', '2_6', '2_7', '2_8']
    numb_views = len(suffixes)
    to_tensor = transforms.ToTensor()
    mvi = []
    for suff in suffixes:
        vi = Image.open(os.path.join(load_folder, class_name, model_name + '_' + suff + '.jpg'))
        mvi.append(to_tensor(vi).unsqueeze(0))
    mvi = torch.cat(mvi, dim=0).cuda()
    _, _, _, _, mv_cdw = net(mvi)
    mv_cdw = mv_cdw.cpu().data.numpy()
    mv_cdw_expt_path = os.path.join(expt_folder, 'mv_cdw', class_name, model_name + '_' + 'mv_cdw' + '.xyz')
    np.savetxt(mv_cdw_expt_path, mv_cdw, fmt='%.8f', delimiter=' ')


