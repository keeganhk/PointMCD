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
dataset_folder = os.path.join(data_root, 'ShapeNetPart')
snp_classes, snp_class_parts = ShapeNetPart_Info()
snp_class_colors = ShapeNetPart_Colors()
pretr_ckpt_params = os.path.join(ckpt_root, 'teacher', 'CNN_Backbone_Wrapper_Pretrained_ShapeNetPart.pth')
ckpt_params = os.path.join(ckpt_root, 'teacher', 'Teacher_Seg_ShapeNetPart.pth')


# pretraining
net = CNN_Backbone_Pretrainer().cuda()
tr_bs = 32
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
    vi, _, _, _ = ShapeNetPart_Balanced_SVI_Loader(dataset_folder, tr_bs, 'train')    
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
        torch.save(net.state_dict(), pretr_ckpt_params)
        num_samples = 0
        average_loss = 0


# training
net = Teacher_Seg().cuda()
tr_bs = 32 
max_lr = 1e-4
min_lr = 5e-6
tr_itr = 100000
show_every = 500
optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=max_lr, weight_decay=1e-5)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=tr_itr, eta_min=min_lr)
criterion = nn.CrossEntropyLoss()
net.train()
for it in tqdm(range(1, tr_itr+1)):
    num_samples = 0
    average_loss = 0
    vi, vl, class_id, model_name_with_suff = ShapeNetPart_Balanced_SVI_Loader(dataset_folder, tr_bs, 'train')
    vi = vi.cuda()
    vl = vl.long().cuda()
    class_id = class_id.long().cuda()
    class_id_oh = F.one_hot(class_id, 16).float()
    batch_size = len(model_name_with_suff)
    optimizer.zero_grad()      
    cdw, logits = net(vi, class_id_oh)
    preds = logits.max(dim=1, keepdim=True)[1]
    labels_reshaped = vl.squeeze(1).view(-1)
    logits_reshaped = logits.permute(0, 2, 3, 1).contiguous().view(-1, 51)
    loss = criterion(logits_reshaped, labels_reshaped)
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


# testing and exporting knowledge
net = Teacher_Seg().cuda()
net.load_state_dict(torch.load(ckpt_params))
net.eval()
el = [1, 2]
az = [1, 2, 3, 4, 5, 6, 7, 8]
suffixes = []
for el_id in el:
    for az_id in az:
        suffixes.append(str(el_id) + '_' + str(az_id))
to_tensor = transforms.ToTensor()
test_list = [line.strip() for line in open(os.path.join(dataset_folder, 'test_list.txt'), 'r')]
class_list = [line.strip() for line in open(os.path.join(dataset_folder, 'class_list.txt'), 'r')]
num_testing = 0
miou = 0
net.eval()
for model_name in tqdm(test_list):
    class_name = model_name[:-5]
    vi = []
    vl = []
    class_id = []
    model_name_with_suff = []
    for suff in suffixes:
        model_name_with_suff_this = model_name + '_' + suff
        model_name_with_suff.append(model_name_with_suff_this)
        class_id_this = class_list.index(class_name)
        class_id.append(class_id_this)
        load_image_folder = os.path.join(data_root, 'ShapeNetPart', 'mv_image_16', class_name)
        load_label_folder = os.path.join(data_root, 'ShapeNetPart', 'mv_image_labeled_16', class_name)
        vi_this = Image.open(os.path.join(load_image_folder, model_name_with_suff_this + '.jpg'))
        vi_this = to_tensor(vi_this).unsqueeze(0)
        vl_this = Image.open(os.path.join(load_label_folder, model_name_with_suff_this + '.png'))
        vl_this = torch.from_numpy(np.array(vl_this)).unsqueeze(0).unsqueeze(0)
        vi.append(vi_this)
        vl.append(vl_this)
    vi = torch.cat(vi, dim=0)
    vl = torch.cat(vl, dim=0)
    class_id = torch.from_numpy(np.array(class_id))
    with torch.no_grad():
        vi = vi.cuda()
        vl = vl.long().cuda()
        class_id = class_id.long().cuda()
        class_id_oh = F.one_hot(class_id, 16).float()      
        batch_size = len(model_name_with_suff)
        num_testing += batch_size
        cdw, logits = net(vi, class_id_oh)
        preds = logits.max(dim=1, keepdim=True)[1]
        parts = snp_class_parts[class_name].copy()
        parts.append(50)
        for bid in range(batch_size):
            preds_this_visualized = visualize_image_tensor(snp_color_mapping_img(preds[bid]))
            preds_this = preds[bid].cpu().data.numpy()
            labels_this = vl[bid].cpu().data.numpy()
            class_name = snp_classes[class_id[bid]]
            parts_iou = []
            for part_this in parts:
                if (labels_this==part_this).sum() == 0:
                    parts_iou.append(1.0)
                else:
                    I = np.sum(np.logical_and(preds_this==part_this, labels_this==part_this))
                    U = np.sum(np.logical_or(preds_this==part_this, labels_this==part_this))
                    parts_iou.append(float(I) / float(U))
            iou_this = np.array(parts_iou).mean()
            miou += iou_this
miou /= num_testing
print('miou: {}%'.format(np.around(miou*100, 2)))


