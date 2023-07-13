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
dataset_folder = os.path.join(data_root, 'ModelNet40')
ckpt_params = os.path.join(ckpt_root, 'teacher', 'Teacher_Cls_ModelNet40.pth')


# training
tr_bs = 48
tr_set = ModelNet40_MVI_Loader(root=dataset_folder, mode='train')
tr_loader = DataLoader(tr_set, batch_size=tr_bs, shuffle=True, num_workers=8, worker_init_fn=seed_worker, drop_last=True)
te_bs = 48
te_set = ModelNet40_MVI_Loader(root=dataset_folder, mode='test')
te_loader = DataLoader(te_set, batch_size=te_bs, shuffle=False, num_workers=8, worker_init_fn=seed_worker, drop_last=False)
net = Teacher_Cls().cuda()
max_lr = 1e-4
min_lr = 1e-6
train_epoch = 128
optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=max_lr, weight_decay=1e-5)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_epoch, eta_min=min_lr)
best_test_acc = 0
for epc in range(1, train_epoch+1):
    net.train()
    epoch_loss = 0
    num_samples = 0
    num_correct = 0
    for (mvi, cid, model_name) in tqdm(tr_loader):
        mvi = mvi.cuda()
        cid = cid.long().cuda()
        bs, nv = mvi.size(0), mvi.size(1)
        optimizer.zero_grad()
        mv_cdw, cdw, mv_lgt, lgt = net(mvi)
        loss_main = smooth_cross_entropy(lgt, cid, eps=0.2)
        loss_auxi = smooth_cross_entropy(mv_lgt.view(bs*nv, -1), cid.unsqueeze(-1).repeat(1, nv).view(-1), eps=0.2)
        loss = loss_main + loss_auxi
        epoch_loss += loss.item() * bs
        prd = lgt.max(dim=-1)[1]
        num_samples += bs
        num_correct += (prd==cid).sum().item()
        loss.backward()
        optimizer.step()
    scheduler.step()
    epoch_loss = np.around(epoch_loss/num_samples, 6)
    train_acc = np.around((num_correct/num_samples)*100, 2)
    net.eval()
    num_samples = 0
    num_correct = 0
    for (mvi, cid, model_name) in tqdm(te_loader):
        with torch.no_grad():
            mvi = mvi.cuda()
            cid = cid.long().cuda()
            bs, nv = mvi.size(0), mvi.size(1)
            mv_cdw, cdw, mv_lgt, lgt = net(mvi)
            prd = lgt.max(dim=-1)[1]
            num_samples += bs
            num_correct += (prd==cid).sum().item()  
    test_acc = np.around((num_correct/num_samples)*100, 2)
    if test_acc > best_test_acc:
        best_test_acc = test_acc
        torch.save(net.state_dict(), ckpt_params)
    print('epc: {}, tr acc: {}%,  te acc: {}%, best te acc: {}%'.format(align_number(epc, 3), train_acc, test_acc, best_test_acc))


# testing and exporting knowledge
te_bs = 32
te_set = ModelNet40_MVI_Loader(root=dataset_folder, mode='test')
te_loader = DataLoader(te_set, batch_size=te_bs, shuffle=False, num_workers=8, worker_init_fn=seed_worker, drop_last=False)
net = Teacher_Cls().cuda()
net.load_state_dict(torch.load(ckpt_params))
net.eval()
num_samples = 0
num_correct = 0
for (mvi, cid, model_name) in tqdm(te_loader):
    with torch.no_grad():
        mvi = mvi.cuda()
        cid = cid.long().cuda()
        bs, nv = mvi.size(0), mvi.size(1)
        mv_cdw, cdw, mv_lgt, lgt = net(mvi)
        prd = lgt.max(dim=-1)[1]
        num_samples += bs
        num_correct += (prd==cid).sum().item()
test_acc = np.around((num_correct/num_samples)*100, 2)
print('oa: {}%'.format(test_acc))
train_model_list = [line.strip() for line in open(os.path.join(dataset_folder, 'train_list.txt'), 'r')]
load_folder = os.path.join(dataset_folder, 'mv_image_12')
expt_folder = os.path.join(expt_root, 'teacher_cls', 'ModelNet40')
for model_name in tqdm(train_model_list):
    class_name = model_name[:-5]
    suffixes = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
    numb_views = len(suffixes)
    to_tensor = transforms.ToTensor()
    mvi = []
    for suff in suffixes:
        vi = Image.open(os.path.join(load_folder, class_name, model_name + '_' + suff + '.jpg'))
        mvi.append(to_tensor(vi).unsqueeze(0))
    mvi = torch.cat(mvi, dim=0).unsqueeze(0).cuda()
    mv_cdw, cdw, mv_lgt, lgt = net(mvi)
    mv_cdw = mv_cdw.cpu().data.numpy()
    cdw = cdw.cpu().data.numpy()
    mv_lgt = mv_lgt.cpu().data.numpy()
    lgt = lgt.cpu().data.numpy()
    cdw_expt_path = os.path.join(expt_folder, 'cdw', class_name, model_name + '_' + 'cdw' + '.xyz')
    lgt_expt_path = os.path.join(expt_folder, 'lgt', class_name, model_name + '_' + 'lgt' + '.xyz')
    mv_cdw_expt_path = os.path.join(expt_folder, 'mv_cdw', class_name, model_name + '_' + 'mv_cdw' + '.xyz')
    mv_lgt_expt_path = os.path.join(expt_folder, 'mv_lgt', class_name, model_name + '_' + 'mv_lgt' + '.xyz')
    np.savetxt(cdw_expt_path, cdw, fmt='%.8f', delimiter=' ')
    np.savetxt(lgt_expt_path, lgt, fmt='%.8f', delimiter=' ')
    np.savetxt(mv_cdw_expt_path, mv_cdw[0], fmt='%.8f', delimiter=' ')
    np.savetxt(mv_lgt_expt_path, mv_lgt[0], fmt='%.8f', delimiter=' ')


