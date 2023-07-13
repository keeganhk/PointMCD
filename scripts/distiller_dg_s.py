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
snp_classes, snp_class_parts = ShapeNetPart_Info()
snp_class_colors = ShapeNetPart_Colors()
ckpt_params = os.path.join(ckpt_root, 'student', 'DGCNN_Seg_Distiller_ShapeNetPart.pth')


# training
tr_bs = 32
tr_set = ShapeNetPart_Joint_Loader(os.path.join(data_root, 'ShapeNetPart'), os.path.join(expt_root, 'teacher_seg', 'ShapeNetPart'), 'train')
tr_loader = DataLoader(tr_set, batch_size=tr_bs, shuffle=True, num_workers=8, worker_init_fn=seed_worker, drop_last=True)
te_bs = 32
te_set = ShapeNetPart_Joint_Loader(os.path.join(data_root, 'ShapeNetPart'), os.path.join(expt_root, 'teacher_seg', 'ShapeNetPart'), 'test')
te_loader = DataLoader(te_set, batch_size=te_bs, shuffle=False, num_workers=8, worker_init_fn=seed_worker, drop_last=False)
net = DGCNN_Seg_Distiller().cuda()
max_lr = 1e-1
min_lr = 1e-3
train_epoch = 250
optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=max_lr, momentum=0.9, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_epoch, eta_min=min_lr)
criterion = nn.L1Loss()
num_classes = 16
num_parts = 50
num_views = 16
best_test_im_iou = 0
for epc in range(1, train_epoch+1):
    epoch_loss = 0
    num_training = 0
    net.train()
    for (points, labels, mvv, class_id, mv_cdw_t, model_name) in tqdm(tr_loader):
        points = points.cuda()
        labels = labels.long().cuda()
        mvv = mvv.float().cuda()
        class_id = class_id.long().cuda()
        class_id_oh = F.one_hot(class_id, num_classes).float()
        mv_cdw_t = mv_cdw_t.cuda()
        batch_size, num_points = points.size(0), points.size(1)
        optimizer.zero_grad()
        mv_cdw_s, logits = net(points, mvv, class_id_oh)
        loss_task = smooth_cross_entropy(logits.view(batch_size*num_points, num_parts), labels.view(batch_size*num_points))
        loss_dist = criterion(mv_cdw_s, mv_cdw_t)
        loss = 1.0*loss_task + 1.0*loss_dist
        loss.backward()
        optimizer.step()
        pred = logits.max(dim=-1)[1]
        epoch_loss += loss.item() * batch_size
        num_training += batch_size
    scheduler.step()
    epoch_loss = np.around(epoch_loss/num_training, 6)
    if epc >= 150:
        net.eval()
        num_testing = 0
        name_list = []
        iou_list = []
        for (points, labels, class_id, model_name) in tqdm(te_loader):
            with torch.no_grad():
                points = points.cuda()
                labels = labels.long().cuda()
                class_id = class_id.long().cuda()
                class_id_oh = F.one_hot(class_id, num_classes).float()
                batch_size, num_points = points.size(0), points.size(1)
                placeholder_mvv = torch.empty(batch_size, num_points, num_views).cuda()
                _, logits = net(points, placeholder_mvv, class_id_oh)
                preds = logits.max(dim=-1)[1]
                num_testing += batch_size
                for bid in range(batch_size):
                    name_list.append(model_name[bid])
                    preds_this = preds[bid].cpu().data.numpy()
                    labels_this = labels[bid].cpu().data.numpy()
                    class_name = model_name[bid][:-5]
                    parts = snp_class_parts[class_name]
                    parts_iou = []
                    for part_this in parts:
                        if (labels_this==part_this).sum() == 0:
                            parts_iou.append(1.0)
                        else:
                            I = np.sum(np.logical_and(preds_this==part_this, labels_this==part_this))
                            U = np.sum(np.logical_or(preds_this==part_this, labels_this==part_this))
                            parts_iou.append(float(I) / float(U))
                    iou_this = np.array(parts_iou).mean()
                    iou_list.append(iou_this)
        test_im_iou = np.array(iou_list).mean()
        if test_im_iou > best_test_im_iou:
            best_test_im_iou = test_im_iou
            torch.save(net.state_dict(), ckpt_params)
        print('epoch: {}, train_loss: {}, test_im_iou: {}, best_test_im_iou: {}'.format(
        align_number(epc, 3), epoch_loss, np.around(test_im_iou*100, 2), np.around(best_test_im_iou*100, 2)))


# testing
te_bs = 32
te_set = ShapeNetPart_Joint_Loader(os.path.join(data_root, 'ShapeNetPart'), os.path.join(expt_root, 'teacher_seg', 'ShapeNetPart'), 'test')
te_loader = DataLoader(te_set, batch_size=te_bs, shuffle=False, num_workers=8, worker_init_fn=seed_worker, drop_last=False)
net = DGCNN_Seg_Distiller().cuda()
net.load_state_dict(torch.load(ckpt_params))
net.eval()
num_classes = 16
num_parts = 50
num_views = 16
num_testing = 0
name_list = []
iou_list = []
for (points, labels, class_id, model_name) in tqdm(te_loader):
    with torch.no_grad():
        points = points.cuda()
        labels = labels.long().cuda()
        class_id = class_id.long().cuda()
        class_id_oh = F.one_hot(class_id, num_classes).float()
        batch_size, num_points = points.size(0), points.size(1)
        placeholder_mvv = torch.empty(batch_size, num_points, num_views).cuda()
        _, logits = net(points, placeholder_mvv, class_id_oh)
        preds = logits.max(dim=-1)[1]
        num_testing += batch_size
        for bid in range(batch_size):
            name_list.append(model_name[bid])
            preds_this = preds[bid].cpu().data.numpy()
            labels_this = labels[bid].cpu().data.numpy()
            class_name = model_name[bid][:-5]
            parts = snp_class_parts[class_name]
            parts_iou = []
            for part_this in parts:
                if (labels_this==part_this).sum() == 0:
                    parts_iou.append(1.0)
                else:
                    I = np.sum(np.logical_and(preds_this==part_this, labels_this==part_this))
                    U = np.sum(np.logical_or(preds_this==part_this, labels_this==part_this))
                    parts_iou.append(float(I) / float(U))
            iou_this = np.array(parts_iou).mean()
            iou_list.append(iou_this)
test_im_iou = np.array(iou_list).mean()
print('miou: {}%'.format(int(test_im_iou*1000)/10))


