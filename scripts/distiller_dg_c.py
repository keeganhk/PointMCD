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
ckpt_params = os.path.join(ckpt_root, 'student', 'DGCNN_Cls_Distiller_ModelNet40.pth')


# training
tr_bs = 24
tr_set = ModelNet40_Joint_Loader(os.path.join(data_root, 'ModelNet40'), os.path.join(expt_root, 'teacher_cls', 'ModelNet40'), 'train')
tr_loader = DataLoader(tr_set, batch_size=tr_bs, shuffle=True, num_workers=8, worker_init_fn=seed_worker, drop_last=True)
te_bs = 24
te_set = ModelNet40_Joint_Loader(os.path.join(data_root, 'ModelNet40'), os.path.join(expt_root, 'teacher_cls', 'ModelNet40'), 'test')
te_loader = DataLoader(te_set, batch_size=te_bs, shuffle=False, num_workers=8, worker_init_fn=seed_worker, drop_last=False)
net = DGCNN_Cls_Distiller().cuda()
max_lr = 1e-1
min_lr = 1e-3
train_epoch = 350
optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=max_lr, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_epoch, eta_min=min_lr)
criterion = nn.L1Loss()
best_test_acc = 0
for epc in range(1, train_epoch+1):
    net.train()
    losses = [0, 0]
    num_samples = 0
    num_correct = 0
    for (pts, mvv, cid, cdw_t, mv_cdw_t, lgt_t, mv_lgt_t, model_name) in tqdm(tr_loader):
        pts = pts.cuda()
        mvv = mvv.float().cuda()
        cid = cid.long().cuda()
        cdw_t = cdw_t.cuda()
        mv_cdw_t = mv_cdw_t.cuda()
        lgt_t = lgt_t.cuda()
        mv_lgt_t = mv_lgt_t.cuda()
        bs, num_pts, num_views = pts.size(0), pts.size(1), mvv.size(2)
        optimizer.zero_grad()
        cdw_s, mv_cdw_s, lgt_s, mv_lgt_s = net(pts, mvv)
        loss_task = smooth_cross_entropy(lgt_s, cid)
        loss_dist_cdw_mv = criterion(mv_cdw_s, mv_cdw_t)
        loss = 0.1 * loss_task + 1.0 * loss_dist_cdw_mv
        loss.backward()
        optimizer.step()
        prd = lgt_s.max(dim=-1)[1]
        num_correct += (prd==cid).sum().item()
        num_samples += bs
        losses[0] += (bs * loss_task.item())
        losses[1] += (bs * loss_dist_cdw_mv.item() / num_views)
    scheduler.step()
    losses[0] = np.around(losses[0]/num_samples, 6)
    losses[1] = np.around(losses[1]/num_samples, 6)
    train_acc = np.around((num_correct/num_samples)*100, 2)
    if epc >= 200:
        tk_dim = 512
        num_views = 12
        net.eval()
        num_samples = 0
        num_correct = 0
        for (pts, cid, model_name) in tqdm(te_loader):    
            with torch.no_grad():
                pts = pts.cuda()
                cid = cid.long().cuda()
                bs, num_pts = pts.size(0), pts.size(1)
                placeholder_mvv = torch.empty(bs, num_pts, num_views).cuda()
                cdw_s, mv_cdw_s, lgt_s, mv_lgt_s = net(pts, placeholder_mvv)
                prd = lgt_s.max(dim=-1)[1]
                num_correct += (prd==cid).sum().item()
                num_samples += bs
        test_acc = np.around((num_correct/num_samples)*100, 2)
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(net.state_dict(), ckpt_params)
        print('epoch: {}, loss_task: {}, loss_task_mv: {}, train acc: {}%, test acc: {}%, best test acc: {}%'
        .format(align_number(epc, 3), losses[0], losses[1], train_acc, test_acc, best_test_acc))


# testing
te_bs = 64
te_set = ModelNet40_Joint_Loader(os.path.join(data_root, 'ModelNet40'), os.path.join(expt_root, 'teacher_cls', 'ModelNet40'), 'test')
te_loader = DataLoader(te_set, batch_size=te_bs, shuffle=False, num_workers=8, worker_init_fn=seed_worker, drop_last=False)
net = DGCNN_Cls_Distiller().cuda()
net.load_state_dict(torch.load(ckpt_params))
net.eval()
tk_dim = 512
num_views = 12
num_samples = 0
num_correct = 0
for (pts, cid, model_name) in tqdm(te_loader):    
    with torch.no_grad():
        pts = pts.cuda()
        cid = cid.long().cuda()
        bs, num_pts = pts.size(0), pts.size(1)
        placeholder_mvv = torch.empty(bs, num_pts, num_views).cuda()
        cdw_s, mv_cdw_s, lgt_s, mv_lgt_s = net(pts, placeholder_mvv)
        prd = lgt_s.max(dim=-1)[1]
        num_correct += (prd==cid).sum().item()
        num_samples += bs
test_acc = np.around((num_correct/num_samples)*100, 2)
print('oa: {}%'.format(test_acc))


