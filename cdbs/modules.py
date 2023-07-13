from .pkgs import *
from .tools import *
from .miscs import *



class SMLP(nn.Module):
    def __init__(self, ic, oc, is_bn, nl, slope=None):
        super(SMLP, self).__init__()
        assert isinstance(is_bn, bool)
        assert nl in ['none', 'relu', 'leakyrelu', 'tanh', 'sigmoid']
        if nl == 'leakyrelu':
            assert slope is not None
        if slope is not None:
            assert nl == 'leakyrelu'
            assert slope>=0 and slope<=1
        self.is_bn = is_bn
        self.nl = nl
        self.conv = nn.Conv2d(in_channels=ic, out_channels=oc, kernel_size=1, bias=False)
        if self.is_bn:
            self.bn = nn.BatchNorm2d(oc)
        if nl == 'relu':
            self.activate = nn.ReLU(inplace=True)
        if nl == 'leakyrelu':
            self.activate = nn.LeakyReLU(negative_slope=slope, inplace=True)
        if nl == 'tanh':
            self.activate = nn.Tanh()
        if nl == 'sigmoid':
            self.activate = nn.Sigmoid()
    def forward(self, x):
        x = x.permute(0, 2, 1).contiguous().unsqueeze(-1)
        y = self.conv(x)
        if self.is_bn:
            y = self.bn(y)
        if self.nl != 'none':
            y = self.activate(y)   
        y = y.squeeze(-1).permute(0, 2, 1).contiguous()
        return y


class CU(nn.Module):
    def __init__(self, ic, oc, ks, is_bn, nl, slope=None, pad='zeros'):
        super(CU, self).__init__()
        assert np.mod(ks + 1, 2) == 0
        assert isinstance(is_bn, bool)
        assert nl in ['none', 'relu', 'leakyrelu', 'tanh', 'sigmoid']
        if nl == 'leakyrelu':
            assert slope is not None
        if slope is not None:
            assert nl == 'leakyrelu'
            assert slope>=0 and slope<=1
        assert pad in ['zeros', 'reflect', 'replicate', 'circular']
        self.is_bn = is_bn
        self.nl = nl
        self.conv = nn.Conv2d(in_channels=ic, out_channels=oc, kernel_size=ks, stride=1, 
                    padding=(ks-1)//2, dilation=1, groups=1, bias=False, padding_mode=pad)
        if self.is_bn:
            self.bn = nn.BatchNorm2d(oc)
        if nl == 'relu':
            self.activate = nn.ReLU(inplace=True)
        if nl == 'leakyrelu':
            self.activate = nn.LeakyReLU(negative_slope=slope, inplace=True)
        if nl == 'tanh':
            self.activate = nn.Tanh()
        if nl == 'sigmoid':
            self.activate = nn.Sigmoid()
    def forward(self, x):
        y = self.conv(x)
        if self.is_bn:
            y = self.bn(y)
        if self.nl != 'none':
            y = self.activate(y)
        return y


class FC(nn.Module):
    def __init__(self, ic, oc, is_bn, nl, slope=None):
        super(FC, self).__init__()
        assert isinstance(is_bn, bool)
        assert nl in ['none', 'relu', 'leakyrelu', 'tanh', 'sigmoid']
        if nl == 'leakyrelu':
            assert slope is not None
        if slope is not None:
            assert nl == 'leakyrelu'
            assert slope>=0 and slope<=1
        self.is_bn = is_bn
        self.nl = nl
        self.linear = nn.Linear(ic, oc, bias=False)
        if self.is_bn:
            self.bn = nn.BatchNorm1d(oc)
        if nl == 'relu':
            self.activate = nn.ReLU(inplace=True)
        if nl == 'leakyrelu':
            self.activate = nn.LeakyReLU(negative_slope=slope, inplace=True)
        if nl == 'tanh':
            self.activate = nn.Tanh()
        if nl == 'sigmoid':
            self.activate = nn.Sigmoid()
    def forward(self, x):
        y = self.linear(x)
        if self.is_bn:
            y = self.bn(y)
        if self.nl != 'none':
            y = self.activate(y)
        return y


class Spatial_Attention(nn.Module):
    def __init__(self, conv_ks):
        super(Spatial_Attention, self).__init__()
        assert conv_ks>=3 and np.mod(conv_ks+1, 2)==0
        self.conv = CU(2, 1, conv_ks, False, 'sigmoid')
    def forward(self, x):
        avg_pooled = torch.mean(x, dim=1, keepdim=True)
        max_pooled = torch.max(x, dim=1, keepdim=True)[0]
        cat_pooled = torch.cat((avg_pooled, max_pooled), dim=1)
        weights = self.conv(cat_pooled)
        y = x * weights + x
        return y


class Channel_Attention(nn.Module): 
    def __init__(self, in_channels, squeeze_ratio):
        super(Channel_Attention, self).__init__()
        inter_channels = in_channels // squeeze_ratio
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        fc_1 = FC(in_channels, inter_channels, False, 'relu')
        fc_2 = FC(inter_channels, in_channels, False, 'none')
        self.fc = nn.Sequential(fc_1, fc_2)
    def forward(self, x):
        avg_pooled = self.avg_pool(x).squeeze(-1).squeeze(-1)
        max_pooled = self.max_pool(x).squeeze(-1).squeeze(-1)
        ap_weights = self.fc(avg_pooled)
        mp_weights = self.fc(max_pooled)
        weights = F.sigmoid(ap_weights + mp_weights)
        y = x * weights.unsqueeze(-1).unsqueeze(-1) + x
        return y


class STN3d(nn.Module):
    def __init__(self, channel):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.fc1 = nn.Linear(256, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = torch.nn.Conv1d(64, 64, 1)
        self.fc2 = nn.Linear(64, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = torch.nn.Conv1d(64, 256, 1)
        self.fc3 = nn.Linear(64, 9)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(64)
        self.bn5 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 256)
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        iden = torch.tensor(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.k = k
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.fc1 = nn.Linear(256, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = torch.nn.Conv1d(64, 64, 1)
        self.fc2 = nn.Linear(64, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = torch.nn.Conv1d(64, 256, 1)
        self.fc3 = nn.Linear(64, k * k)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(64)
        self.bn5 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 256)
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        iden = torch.tensor(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).repeat(batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


def feature_transform_reguliarzer(trans):
    d = trans.size()[1]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss


class PointNet_Encoder(nn.Module):
    def __init__(self, feature_transform, channel):
        super(PointNet_Encoder, self).__init__()
        self.feature_transform = feature_transform 
        self.stn = STN3d(channel)
        if self.feature_transform:
            self.fstn = STNkd(k=64) 
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn3 = nn.BatchNorm1d(1024)
    def forward(self, x):
        B, D, N = x.size()
        trans = self.stn(x)
        x = x.transpose(2, 1)
        if D > 3:
            feature = x[:, :, 3:]
            x = x[:, :, :3]
        x = torch.bmm(x, trans)
        if D > 3:
            x = torch.cat([x, feature], dim=2)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None
        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return x, trans, trans_feat


class Student_PointNet(nn.Module):
    def __init__(self):
        super(Student_PointNet, self).__init__()
        self.encoder = PointNet_Encoder(True, 3)
    def forward(self, pts):
        pwe, trans, trans_feat = self.encoder(pts.permute(0, 2, 1))
        pwe = pwe.permute(0, 2, 1).contiguous()
        loss_rglz = feature_transform_reguliarzer(trans_feat)
        return pwe, loss_rglz


class EdgeConv(nn.Module):
    def __init__(self, in_channels, out_channels, num_neighbors, num_layers):
        super(EdgeConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_neighbors = num_neighbors
        self.num_layers = num_layers
        assert num_layers in [1, 2]
        if self.num_layers == 1:
            self.smlp = SMLP(in_channels*2, out_channels, is_bn=True, nl='leakyrelu', slope=0.20)
        if self.num_layers == 2:
            smlp_1 = SMLP(in_channels*2, out_channels, is_bn=True, nl='leakyrelu', slope=0.20)
            smlp_2 = SMLP(out_channels, out_channels, is_bn=True, nl='leakyrelu', slope=0.20)
            self.smlp = nn.Sequential(smlp_1, smlp_2)
    def forward(self, pc_ftr):
        num_neighbors = self.num_neighbors
        batch_size, num_points, in_channels = pc_ftr.size()
        knn_indices = knn_search(pc_ftr.detach(), pc_ftr.detach(), num_neighbors)
        nb_ftr = index_points(pc_ftr, knn_indices)
        pc_ftr_rep = pc_ftr.unsqueeze(2).repeat(1, 1, num_neighbors, 1)
        edge_ftr = torch.cat((pc_ftr_rep, nb_ftr-pc_ftr_rep), dim=-1)
        out_ftr = self.smlp(edge_ftr.view(batch_size, num_points*num_neighbors, -1)).view(batch_size, num_points, num_neighbors, -1)
        out_ftr_max_pooled = torch.max(out_ftr, dim=2)[0]
        return out_ftr_max_pooled


class Student_DGCNN_Cls(nn.Module):
    def __init__(self, in_channels=3, out_channels=1024):
        super(Student_DGCNN_Cls, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        hidden_neurons = [64, 64, 128, 256]
        self.econv_1 = EdgeConv(in_channels, hidden_neurons[0], 20, 1)
        self.econv_2 = EdgeConv(hidden_neurons[0], hidden_neurons[1], 20, 1)
        self.econv_3 = EdgeConv(hidden_neurons[1], hidden_neurons[2], 20, 1)
        self.econv_4 = EdgeConv(hidden_neurons[2], hidden_neurons[3], 20, 1)
        cat_channels = int(np.array(hidden_neurons).sum())
        self.multi_scale_fusion = SMLP(cat_channels, out_channels, is_bn=True, nl='leakyrelu', slope=0.20)
        fc_dim = [2048, 512, 256, 40]
        fc_1 = nn.Sequential(nn.Linear(fc_dim[0], fc_dim[1], bias=False), nn.BatchNorm1d(fc_dim[1]), nn.LeakyReLU(True, 0.20), nn.Dropout(0.5))
        fc_2 = nn.Sequential(nn.Linear(fc_dim[1], fc_dim[2], bias=False), nn.BatchNorm1d(fc_dim[2]), nn.LeakyReLU(True, 0.20), nn.Dropout(0.5))
        fc_3 = nn.Sequential(nn.Linear(fc_dim[2], fc_dim[3], bias=False))
        self.fc = nn.Sequential(fc_1, fc_2, fc_3)
    def forward(self, pc):
        batch_size, num_points = pc.size(0), pc.size(1)
        f1 = self.econv_1(pc)
        f2 = self.econv_2(f1)
        f3 = self.econv_3(f2)
        f4 = self.econv_4(f3)
        ftr_cat = torch.cat((f1, f2, f3, f4), dim=-1)
        pw_ftr = self.multi_scale_fusion(ftr_cat)
        cdw = torch.cat((pw_ftr.max(dim=1)[0], pw_ftr.mean(dim=1)), dim=-1)
        lgt = self.fc(cdw)
        return pw_ftr, cdw, lgt

    
class Student_DGCNN_Seg(nn.Module):
    def __init__(self):
        super(Student_DGCNN_Seg, self).__init__()
        self.econv_1 = EdgeConv(3, 64, 40, 2)
        self.econv_2 = EdgeConv(64, 64, 40, 2)
        self.econv_3 = EdgeConv(64, 64, 40, 1)
        self.pw_embedding = SMLP(192, 1024, True, 'relu')
        self.class_lifting = FC(16, 64, True, 'leakyrelu', 0.20)
        self.seg_conv_1 = SMLP(1280, 256, True, 'leakyrelu', 0.20)
        self.dp_1 = nn.Dropout(p=0.5)
        self.seg_conv_2 = SMLP(256, 256, True, 'leakyrelu', 0.20)
        self.dp_2 = nn.Dropout(p=0.5)
        self.seg_conv_3 = SMLP(256, 128, True, 'leakyrelu', 0.20)
        self.seg_conv_4 = SMLP(128, 50, False, 'none')
    def forward(self, points, class_id_oh):
        num_points = points.size(1)
        pw_ftr_1 = self.econv_1(points)
        pw_ftr_2 = self.econv_2(pw_ftr_1)
        pw_ftr_3 = self.econv_3(pw_ftr_2)
        pw_ftr_cat = torch.cat((pw_ftr_1, pw_ftr_2, pw_ftr_3), dim=-1)
        pw_ftr = self.pw_embedding(pw_ftr_cat)
        codeword = torch.max(pw_ftr, dim=1)[0]
        categorical_vector = self.class_lifting(class_id_oh)
        codeword_guided = torch.cat((codeword, categorical_vector), dim=-1)
        codeword_guided_dup = codeword_guided.unsqueeze(1).repeat(1, num_points, 1)
        pw_ftr_fused = torch.cat((codeword_guided_dup, pw_ftr_1, pw_ftr_2, pw_ftr_3), dim=-1)
        seg_ftr_1 = self.dp_1(self.seg_conv_1(pw_ftr_fused))
        seg_ftr_2 = self.dp_2(self.seg_conv_2(seg_ftr_1))
        seg_ftr_3 = self.seg_conv_3(seg_ftr_2)
        lgt = self.seg_conv_4(seg_ftr_3)
        return pw_ftr, codeword, lgt


class CNN_Backbone(nn.Module):
    def __init__(self):
        super(CNN_Backbone, self).__init__()
        bb = torchvision.models.vgg11_bn(pretrained=True).features
        vgg_conv_1 = nn.Sequential(bb[0], bb[1], bb[2], bb[3], bb[4], bb[5], bb[6])
        spat_att_1 = Spatial_Attention(7)
        vgg_conv_2 = nn.Sequential(bb[7], bb[8], bb[9], bb[10], bb[11], bb[12], bb[13])
        spat_att_2 = Spatial_Attention(5)
        vgg_conv_3 = nn.Sequential(bb[14], bb[15], bb[16], bb[17], bb[18], bb[19], bb[20])
        chan_att_3 = Channel_Attention(512, 8)
        vgg_conv_4 = nn.Sequential(bb[21], bb[22], bb[23], bb[24], bb[25], bb[26], bb[27])
        chan_att_4 = Channel_Attention(512, 8)
        self.conv_1 = nn.Sequential(vgg_conv_1, spat_att_1) 
        self.conv_2 = nn.Sequential(vgg_conv_2, spat_att_2) 
        self.conv_3 = nn.Sequential(vgg_conv_3, chan_att_3)
        self.conv_4 = nn.Sequential(vgg_conv_4, chan_att_4)
        self.pool_4 = nn.AdaptiveAvgPool2d(1)
    def forward(self, image):
        ftr_1 = self.conv_1(image)
        ftr_2 = self.conv_2(ftr_1)
        ftr_3 = self.conv_3(ftr_2)
        ftr_4 = self.conv_4(ftr_3)
        cdw = self.pool_4(ftr_4).squeeze(-1).squeeze(-1)
        return ftr_1, ftr_2, ftr_3, ftr_4, cdw


class CNN_Backbone_Pretrainer(nn.Module):
    def __init__(self):
        super(CNN_Backbone_Pretrainer, self).__init__()
        self.cnn_backbone = CNN_Backbone()
        self.lifting = FC(512, 7*7*256, True, 'relu')
        self.ups_x2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.deconv_1 = CU(256, 256, 3, True, 'relu')
        self.deconv_2 = CU(256, 128, 3, True, 'relu')
        self.deconv_3 = CU(128, 64, 3, True, 'relu')
        self.deconv_4 = CU(64, 32, 3, True, 'relu')
        self.output = CU(32, 3, 3, False, 'none')
    def forward(self, image):
        batch_size = image.size(0)
        ftr_1, ftr_2, ftr_3, ftr_4, cdw = self.cnn_backbone(image)
        rec_0 = self.ups_x2(self.lifting(cdw).view(batch_size, 256, 7, 7))
        rec_1 = self.ups_x2(self.deconv_1(rec_0))
        rec_2 = self.ups_x2(self.deconv_2(rec_1))
        rec_3 = self.ups_x2(self.deconv_3(rec_2))
        rec_4 = self.ups_x2(self.deconv_4(rec_3))
        rec_image = self.output(rec_4)
        return rec_image


def build_cnn_backbone_pretrainer(dataset, ckpt_folder):
    assert dataset in ['ShapeNetCore', 'ShapeNetPart']
    cnn_backbone_pretrainer = CNN_Backbone_Pretrainer()
    if dataset == 'ShapeNetCore':
        pretrained_params = torch.load(os.path.join(ckpt_folder, 'CNN_Backbone_Wrapper_Pretrained_ShapeNetCore.pth'))
        cnn_backbone_pretrainer.load_state_dict(pretrained_params)
    if dataset == 'ShapeNetPart':
        pretrained_params = torch.load(os.path.join(ckpt_folder, 'CNN_Backbone_Wrapper_Pretrained_ShapeNetPart.pth'))
        cnn_backbone_pretrainer.load_state_dict(pretrained_params)
    return cnn_backbone_pretrainer


class Teacher_Cls(nn.Module):
    def __init__(self):
        super(Teacher_Cls, self).__init__()
        backbone_dim = 1280
        codeword_dim = 512
        numb_classes = 40
        self.cnn_backbone = torchvision.models.mobilenet_v2(True).features
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.mv_cdw_embedding = SMLP(backbone_dim, codeword_dim, True, 'relu')
        self.cdw_embedding = FC(codeword_dim, codeword_dim, True, 'relu')
        self.classifier = nn.Linear(codeword_dim, numb_classes, bias=False)
    def forward(self, mvi):
        bs, nv, ic, ih, iw = mvi.size()
        mv_ftr = self.cnn_backbone(mvi.view(bs*nv, ic, ih, iw))
        mv_cdw = self.mv_cdw_embedding(self.gap(mv_ftr).squeeze(-1).squeeze(-1).view(bs, nv, -1))
        cdw = self.cdw_embedding(mv_cdw.max(dim=1)[0])
        mv_lgt = self.classifier(mv_cdw.view(bs*nv, -1)).view(bs, nv, -1)
        lgt = self.classifier(cdw)
        return mv_cdw, cdw, mv_lgt, lgt


class Teacher_Seg(nn.Module):
    def __init__(self):
        super(Teacher_Seg, self).__init__()
        self.num_parts = 50
        self.num_classes = 16
        self.out_channels = self.num_parts + 1
        cnn_backbone_pretrainer = build_cnn_backbone_pretrainer('ShapeNetPart', '../ckpt/teacher')
        self.bb_dim = [128, 256, 512, 512]
        self.conv_1 = cnn_backbone_pretrainer.cnn_backbone.conv_1
        self.conv_2 = cnn_backbone_pretrainer.cnn_backbone.conv_2
        self.conv_3 = cnn_backbone_pretrainer.cnn_backbone.conv_3
        self.conv_4 = cnn_backbone_pretrainer.cnn_backbone.conv_4
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.cv_dim = 64
        self.lift = FC(self.num_classes, self.cv_dim, True, 'relu')
        self.fs_dim = 128
        self.fuse = FC(self.bb_dim[-1]+self.cv_dim, self.fs_dim, True, 'relu')
        self.ups_x2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.dec_1 = CU(self.bb_dim[-1]+self.fs_dim, self.bb_dim[-1], 1, True, 'relu')
        self.dec_2 = CU(self.bb_dim[-1]+self.bb_dim[-2], self.bb_dim[-2], 3, True, 'relu')
        self.dec_3 = CU(self.bb_dim[-2]+self.bb_dim[-3], self.bb_dim[-3], 3, True, 'relu')
        self.dec_4 = CU(self.bb_dim[-3]+self.bb_dim[-4], self.bb_dim[-4], 3, True, 'relu')
        self.dec_5 = CU(self.bb_dim[-4], 64, 3, True, 'relu')
        self.output = CU(64, self.out_channels, 3, False, 'none')
    def forward(self, vi, class_id_oh):
        bs = vi.size(0)
        ic = vi.size(1)
        ih, iw = vi.size(2), vi.size(3)
        nc = class_id_oh.size(1)
        f1 = self.conv_1(vi)
        f2 = self.conv_2(f1)
        f3 = self.conv_3(f2)
        f4 = self.conv_4(f3)
        cdw = self.gap(f4).squeeze(-1).squeeze(-1)
        cat_vec = self.lift(class_id_oh)
        global_info = self.fuse(torch.cat((cdw, cat_vec), dim=-1))
        global_info_dup = global_info.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, f4.size(2), f4.size(3))
        d1 = self.dec_1(torch.cat((f4, global_info_dup), dim=1))
        d2 = self.dec_2(torch.cat((f3, self.ups_x2(d1)), dim=1))
        d3 = self.dec_3(torch.cat((f2, self.ups_x2(d2)), dim=1))
        d4 = self.dec_4(torch.cat((f1, self.ups_x2(d3)), dim=1))
        d5 = self.dec_5(self.ups_x2(d4))
        lgt = self.output(d5)
        return cdw, lgt


