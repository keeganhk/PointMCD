from .pkgs import *
from .tools import *
from .miscs import *
from .modules import *



class PointNet_Cls_Distiller(nn.Module):
    def __init__(self):
        super(PointNet_Cls_Distiller, self).__init__()
        self.num_pts = 1024
        self.ftr_dim = 1024
        self.encoder = Student_PointNet()
        self.tk_dim = 512
        self.num_views = 12
        self.num_classes = 40
        self.cdw_alignment = FC(self.ftr_dim, self.tk_dim, True, 'relu')
        self.mv_cdw_alignment = FC(self.ftr_dim, self.tk_dim, True, 'relu')
        self.classifier = nn.Linear(self.tk_dim, self.num_classes, bias=False)
    def forward(self, pts, mvv):
        bs = pts.size(0)
        num_pts = self.num_pts
        ftr_dim = self.ftr_dim
        tk_dim = self.tk_dim
        num_views = self.num_views
        num_classes = self.num_classes
        pwe, loss_rglz = self.encoder(pts)
        pwe_pooled = pwe.max(dim=1)[0]
        cdw_s = self.cdw_alignment(pwe_pooled)
        mvv_dup = mvv.permute(0, 2, 1).contiguous().unsqueeze(-1).repeat(1, 1, 1, ftr_dim)
        pwe_dup = pwe.unsqueeze(1).repeat(1, num_views, 1, 1)
        pwe_masked = pwe_dup * mvv_dup
        pwe_masked_pooled = pwe_masked.max(dim=2)[0]
        mv_cdw_s = self.mv_cdw_alignment(pwe_masked_pooled.view(-1, ftr_dim)).view(bs, num_views, tk_dim)
        lgt_s = self.classifier(cdw_s)
        mv_lgt_s = self.classifier(mv_cdw_s.view(-1, tk_dim)).view(bs, num_views, num_classes)
        return cdw_s, mv_cdw_s, lgt_s, mv_lgt_s, loss_rglz


class PointNet_Seg_Distiller(nn.Module):
    def __init__(self):
        super(PointNet_Seg_Distiller, self).__init__()
        self.num_pts = 2048
        self.ftr_dim = 1024
        self.tk_dim = 512
        self.num_classes = 16
        self.num_views = 16
        self.num_parts = 50
        self.encoder = Student_PointNet()
        cat_dim = self.ftr_dim + self.ftr_dim + self.num_classes
        dec_dim = [cat_dim, 512, 256, 128, self.num_parts]
        self.dec_mlp_1 = SMLP(dec_dim[0], dec_dim[1], True, 'relu')
        self.dec_mlp_2 = SMLP(dec_dim[1], dec_dim[2], True, 'relu')
        self.dec_mlp_3 = SMLP(dec_dim[2], dec_dim[3], True, 'relu')
        self.dec_mlp_4 = SMLP(dec_dim[3], dec_dim[4], False, 'none')
        self.mv_cdw_alignment = FC(self.ftr_dim, self.tk_dim, True, 'relu')
    def forward(self, pts, mvv, class_id_oh):
        bs = pts.size(0)
        num_pts = self.num_pts
        ftr_dim = self.ftr_dim
        tk_dim = self.tk_dim
        num_classes = self.num_classes
        num_views = self.num_views
        num_parts = self.num_parts
        pwe, loss_rglz = self.encoder(pts)
        cdw = pwe.max(dim=1)[0]
        cdw_dup = cdw.unsqueeze(1).repeat(1, num_pts, 1)
        class_id_oh_dup = class_id_oh.unsqueeze(1).repeat(1, num_pts, 1)
        pwe_cat = torch.cat((pwe, cdw_dup, class_id_oh_dup), dim=-1)
        dec_1 = self.dec_mlp_1(pwe_cat)
        dec_2 = self.dec_mlp_2(dec_1)
        dec_3 = self.dec_mlp_3(dec_2)
        lgt_s = self.dec_mlp_4(dec_3)
        mvv_dup = mvv.permute(0, 2, 1).contiguous().unsqueeze(-1).repeat(1, 1, 1, ftr_dim)
        pwe_dup = pwe.unsqueeze(1).repeat(1, num_views, 1, 1)
        pwe_masked = pwe_dup * mvv_dup
        pwe_masked_pooled = pwe_masked.max(dim=2)[0]
        mv_cdw_s = self.mv_cdw_alignment(pwe_masked_pooled.view(-1, ftr_dim)).view(bs, num_views, tk_dim)
        return mv_cdw_s, lgt_s


class DGCNN_Cls_Distiller(nn.Module):
    def __init__(self):
        super(DGCNN_Cls_Distiller, self).__init__()
        self.num_pts = 1024
        self.ftr_dim = 1024
        ec_dim = [3, 64, 64, 128, 256]
        self.edge_conv_1 = EdgeConv(ec_dim[0], ec_dim[1], 20, 1)
        self.edge_conv_2 = EdgeConv(ec_dim[1], ec_dim[2], 20, 1)
        self.edge_conv_3 = EdgeConv(ec_dim[2], ec_dim[3], 20, 1)
        self.edge_conv_4 = EdgeConv(ec_dim[3], ec_dim[4], 20, 1)
        cat_dim = ec_dim[1] + ec_dim[2] + ec_dim[3] + ec_dim[4]
        self.ms_fusion = SMLP(cat_dim, self.ftr_dim, True, 'relu')
        self.tk_dim = 512
        self.num_views = 12
        self.num_classes = 40
        self.cdw_alignment = FC(self.ftr_dim, self.tk_dim, True, 'relu')
        self.mv_cdw_alignment = FC(self.ftr_dim, self.tk_dim, True, 'relu')
        self.classifier = nn.Linear(self.tk_dim, self.num_classes, bias=False)
    def forward(self, pts, mvv):
        bs = pts.size(0)
        num_pts = self.num_pts
        ftr_dim = self.ftr_dim
        tk_dim = self.tk_dim
        num_views = self.num_views
        num_classes = self.num_classes
        pwe_1 = self.edge_conv_1(pts)
        pwe_2 = self.edge_conv_2(pwe_1)
        pwe_3 = self.edge_conv_3(pwe_2)
        pwe_4 = self.edge_conv_4(pwe_3)
        pwe =  self.ms_fusion(torch.cat((pwe_1, pwe_2, pwe_3, pwe_4), dim=-1))
        pwe_pooled = pwe.max(dim=1)[0]
        cdw_s = self.cdw_alignment(pwe_pooled)
        mvv_dup = mvv.permute(0, 2, 1).contiguous().unsqueeze(-1).repeat(1, 1, 1, ftr_dim)
        pwe_dup = pwe.unsqueeze(1).repeat(1, num_views, 1, 1)
        pwe_masked = pwe_dup * mvv_dup
        pwe_masked_pooled = pwe_masked.max(dim=2)[0]
        mv_cdw_s = self.mv_cdw_alignment(pwe_masked_pooled.view(-1, ftr_dim)).view(bs, num_views, tk_dim)
        lgt_s = self.classifier(cdw_s)
        mv_lgt_s = self.classifier(mv_cdw_s.view(-1, tk_dim)).view(bs, num_views, num_classes)
        return cdw_s, mv_cdw_s, lgt_s, mv_lgt_s


class DGCNN_Seg_Distiller(nn.Module):
    def __init__(self):
        super(DGCNN_Seg_Distiller, self).__init__()
        self.num_pts = 2048
        self.ftr_dim = 1024
        self.encoder = Student_DGCNN_Seg()
        self.tk_dim = 512
        self.num_classes = 16
        self.num_views = 16
        self.num_parts = 50
        self.mv_cdw_alignment = FC(self.ftr_dim, self.tk_dim, True, 'relu')
    def forward(self, pts, mvv, class_id_oh):
        bs = pts.size(0)
        num_pts = self.num_pts
        ftr_dim = self.ftr_dim
        tk_dim = self.tk_dim
        num_classes = self.num_classes
        num_views = self.num_views
        num_parts = self.num_parts
        pwe, cdw_s, lgt_s = self.encoder(pts, class_id_oh)
        mvv_dup = mvv.permute(0, 2, 1).contiguous().unsqueeze(-1).repeat(1, 1, 1, ftr_dim)
        pwe_dup = pwe.unsqueeze(1).repeat(1, num_views, 1, 1)
        pwe_masked = pwe_dup * mvv_dup
        pwe_masked_pooled = pwe_masked.max(dim=2)[0]
        mv_cdw_s = self.mv_cdw_alignment(pwe_masked_pooled.view(-1, ftr_dim)).view(bs, num_views, tk_dim)
        return mv_cdw_s, lgt_s


