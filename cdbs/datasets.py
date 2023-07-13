from .pkgs import *
from .tools import *
from .miscs import *



class ModelNet40_PC_Loader(torch.utils.data.Dataset):
    def __init__(self, root, mode):
        self.root = root
        self.mode = mode
        assert mode in ['train', 'test']
        self.class_file = os.path.join(root,'class_list.txt')
        self.list_file = os.path.join(root, mode + '_list.txt')
        self.class_list = [line.strip() for line in open(self.class_file, 'r')]  
        self.model_list = [line.strip() for line in open(self.list_file, 'r')]
        self.num_models = len(self.model_list)
    def __getitem__(self, index):
        np.random.seed()
        model_name = self.model_list[index]
        class_name = model_name[:-5]
        class_id = self.class_list.index(class_name)
        model_file = os.path.join(self.root, 'point_cloud_1024', class_name, model_name + '.xyz')
        points = load_pc(model_file, ' ') # (1024, 3)
        if self.mode == 'train':
            points = random_anisotropic_scaling(points, 2/3, 3/2)
            points = random_translation(points, 0.20)            
        return points, class_id, model_name
    def __len__(self):
        return self.num_models
    
    
class ModelNet40_MVI_Loader(torch.utils.data.Dataset):
    def __init__(self, root, mode):
        self.root = root
        self.mode = mode
        assert mode in ['train', 'test']
        list_file = os.path.join(root, mode + '_list.txt')
        self.model_list = [line.strip() for line in open(list_file, 'r')]
        self.mn_classes = ['airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 'chair', 
        'cone', 'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot', 'glass_box', 'guitar', 'keyboard', 
        'lamp', 'laptop', 'mantel', 'monitor', 'night_stand', 'person', 'piano', 'plant', 'radio', 'range_hood', 
        'sink','sofa', 'stairs', 'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase', 'wardrobe', 'xbox']
        self.suffixes = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
        self.to_tensor = transforms.ToTensor()
    def __getitem__(self, index):
        np.random.seed()
        model_name = self.model_list[index]
        class_name = model_name[:-5]
        class_id = self.mn_classes.index(class_name)
        mvi = []
        for suff in self.suffixes:
            vi = Image.open(os.path.join(self.root, 'mv_image_12', class_name, model_name + '_' + suff + '.jpg'))
            mvi.append(self.to_tensor(vi).unsqueeze(0))
        mvi = torch.cat(mvi, dim=0)
        return mvi, class_id, model_name
    def __len__(self):
        return len(self.model_list)


def ModelNet40_Balanced_SVI_Loader(root, batch_size, running_mode):
    np.random.seed()
    assert running_mode in ['train', 'test', 'total']
    num_classes = 40
    mn_classes = ['airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 'chair', 
    'cone', 'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot', 'glass_box', 'guitar', 'keyboard', 
    'lamp', 'laptop', 'mantel', 'monitor', 'night_stand', 'person', 'piano', 'plant', 'radio', 'range_hood', 
    'sink','sofa', 'stairs', 'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase', 'wardrobe', 'xbox']
    train_class_index_dict = {'airplane': [1, 626], 'bathtub': [1, 106], 'bed': [1, 515], 'bench': [1, 173], 'bookshelf': [1, 572], 
    'bottle': [1, 335], 'bowl': [1, 64], 'car': [1, 197], 'chair': [1, 889], 'cone': [1, 167], 'cup': [1, 79], 'curtain': [1, 138], 
    'desk': [1, 200], 'door': [1, 109], 'dresser': [1, 200], 'flower_pot': [1, 149], 'glass_box': [1, 171], 'guitar': [1, 155], 
    'keyboard': [1, 145], 'lamp': [1, 124], 'laptop': [1, 149], 'mantel': [1, 284], 'monitor': [1, 465], 'night_stand': [1, 200], 
    'person': [1, 88], 'piano': [1, 231], 'plant': [1, 240], 'radio': [1, 104], 'range_hood': [1, 115], 'sink': [1, 128], 
    'sofa': [1, 680], 'stairs': [1, 124], 'stool': [1, 90], 'table': [1, 392], 'tent': [1, 163], 'toilet': [1, 344], 
    'tv_stand': [1, 267], 'vase': [1, 475], 'wardrobe': [1, 87], 'xbox': [1, 103]}
    test_class_index_dict = {'airplane': [627, 726], 'bathtub': [107, 156], 'bed': [516, 615], 'bench': [174, 193],'bookshelf': [573, 672],
    'bottle': [336, 435], 'bowl': [65, 84], 'car': [198, 297], 'chair': [890, 989], 'cone': [168, 187], 'cup': [80, 99], 'curtain': [139, 158], 
    'desk': [201, 286], 'door': [110, 129], 'dresser': [201, 286], 'flower_pot': [150, 169], 'glass_box': [172, 271], 'guitar': [156, 255], 
    'keyboard': [146, 165], 'lamp': [125, 144], 'laptop': [150, 169], 'mantel': [285, 384], 'monitor': [466, 565], 'night_stand': [201, 286], 
    'person': [89, 108], 'piano': [232, 331], 'plant': [241, 340], 'radio': [105, 124], 'range_hood': [116, 215], 'sink': [129, 148],
    'sofa': [681, 780], 'stairs': [125, 144], 'stool': [91, 110], 'table': [393, 492], 'tent': [164, 183], 'toilet': [345, 444], 
    'tv_stand': [268, 367], 'vase': [476, 575], 'wardrobe': [88, 107], 'xbox': [104, 123]}
    total_class_index_dict = {'airplane': [1, 726], 'bathtub': [1, 156], 'bed': [1, 615], 'bench': [1, 193], 'bookshelf': [1, 672], 
    'bottle': [1, 435], 'bowl': [1, 84], 'car': [1, 297], 'chair': [1, 989], 'cone': [1, 187], 'cup': [1, 99], 'curtain': [1, 158], 
    'desk': [1, 286], 'door': [1, 129], 'dresser': [1, 286], 'flower_pot': [1, 169], 'glass_box': [1, 271], 'guitar': [1, 255], 
    'keyboard': [1, 165], 'lamp': [1, 144], 'laptop': [1, 169], 'mantel': [1, 384], 'monitor': [1, 565], 'night_stand': [1, 286], 
    'person': [1, 108], 'piano': [1, 331], 'plant': [1, 340], 'radio': [1, 124], 'range_hood': [1, 215], 'sink': [1, 148], 
    'sofa': [1, 780], 'stairs': [1, 144], 'stool': [1, 110], 'table': [1, 492], 'tent': [1, 183], 'toilet': [1, 444], 
    'tv_stand': [1, 367], 'vase': [1, 575], 'wardrobe': [1, 107], 'xbox': [1, 123]}
    if running_mode == 'total':
        target_class_index_dict = total_class_index_dict
    if running_mode == 'train':
        target_class_index_dict = train_class_index_dict
    if running_mode == 'test':
        target_class_index_dict = test_class_index_dict
    el = [1, 2]
    az = [1, 2, 3, 4, 5, 6, 7, 8]
    suffixes = []
    for el_id in el:
        for az_id in az:
            suffixes.append(str(el_id) + '_' + str(az_id))
    vi = []
    class_id = []
    model_name_with_suff = []
    to_tensor = transforms.Compose([transforms.ToTensor()])
    selected_classes = np.random.choice(num_classes, batch_size)
    for bid in range(batch_size):
        class_id_this = selected_classes[bid]
        class_id.append(class_id_this)
        class_name = mn_classes[class_id_this]
        start_model_index = target_class_index_dict[class_name][0]
        end_model_index = target_class_index_dict[class_name][1]
        selected_model_index = np.random.randint(low=start_model_index, high=end_model_index+1)
        selected_model_suffix = np.random.choice(suffixes)
        model_name_with_suff_this = class_name + '_' + align_number(selected_model_index, 4) + '_' + selected_model_suffix
        model_name_with_suff.append(model_name_with_suff_this)
        view_image = Image.open(os.path.join(root, 'mv_image', class_name, model_name_with_suff_this + '.jpg'))
        view_image = to_tensor(view_image).unsqueeze(0) # [1, 3, img_size, img_size]
        vi.append(view_image)
    vi = torch.cat(vi, dim=0) # [batch_size, 3, img_size, img_size]
    class_id = torch.from_numpy(np.array(class_id))
    return vi, class_id, model_name_with_suff


class ModelNet40_Joint_Loader(torch.utils.data.Dataset):
    def __init__(self, data_root, expt_root, mode):
        self.data_root = data_root
        self.expt_root = expt_root
        self.mode = mode
        assert mode in ['train', 'test']
        self.class_file = os.path.join(data_root,'class_list.txt')
        self.list_file = os.path.join(data_root, mode + '_list.txt')
        self.class_list = [line.strip() for line in open(self.class_file, 'r')]  
        self.model_list = [line.strip() for line in open(self.list_file, 'r')]
        self.num_models = len(self.model_list)
    def __getitem__(self, index):
        np.random.seed()
        model_name = self.model_list[index]
        class_name = model_name[:-5]
        cid = self.class_list.index(class_name)
        # load point cloud
        pts = load_pc(os.path.join(self.data_root, 'point_cloud_1024', class_name, model_name + '.xyz'), ' ')
        if self.mode == 'train':
            # load multi-view visibility
            mvv = load_pc(os.path.join(self.data_root, 'mv_visibility_1024x12', class_name, model_name + '_' + 'mvv' + '.xyz'), ' ')
            # load teacher knowledge 
            cdw_t = load_pc(os.path.join(self.expt_root, 'cdw', class_name, model_name + '_' + 'cdw' + '.xyz'), ' ')
            mv_cdw_t = load_pc(os.path.join(self.expt_root, 'mv_cdw', class_name, model_name + '_' + 'mv_cdw' + '.xyz'), ' ')
            lgt_t = load_pc(os.path.join(self.expt_root, 'lgt', class_name, model_name + '_' + 'lgt' + '.xyz'), ' ')
            mv_lgt_t = load_pc(os.path.join(self.expt_root, 'mv_lgt', class_name, model_name + '_' + 'mv_lgt' + '.xyz'), ' ')
            # point cloud data augmentation
            pts = random_anisotropic_scaling(pts, 2/3, 3/2)
            pts = random_translation(pts, 0.20)
            return pts, mvv, cid, cdw_t, mv_cdw_t, lgt_t, mv_lgt_t, model_name
        if self.mode == 'test':
            return pts, cid, model_name
    def __len__(self):
        return self.num_models


def ShapeNetPart_Info():
    snp_classes = ['airplane', 'bag', 'cap', 'car', 'chair', 'earphone', 'guitar', 'knife', 
    'lamp', 'laptop', 'motorbike', 'mug', 'pistol', 'rocket', 'skateboard', 'table']
    snp_class_parts = {'airplane': [0, 1, 2, 3], 'bag': [4, 5], 'cap': [6, 7], 'car': [8, 9, 10, 11],
    'chair': [12, 13, 14, 15], 'earphone': [16, 17, 18], 'guitar': [19, 20, 21], 'knife': [22, 23],
    'lamp': [24, 25, 26, 27], 'laptop': [28, 29], 'motorbike': [30, 31, 32, 33, 34, 35], 'mug': [36, 37],
    'pistol': [38, 39, 40], 'rocket': [41, 42, 43], 'skateboard': [44, 45, 46], 'table': [47, 48, 49]}
    return snp_classes, snp_class_parts


def ShapeNetPart_Colors():
    # snp_class_colors: (50, 3)
    _, snp_class_parts = ShapeNetPart_Info()
    num_parts = []
    for k, v in snp_class_parts.items():
        num_parts.append(len(v))
    cmap = cm.jet
    base_colors = cmap(np.linspace(0, 1, 6))[:, 0:3] * 255
    snp_class_colors = np.zeros((50, 3))
    i = 0
    for num in num_parts:
        for k in range(num):
            snp_class_colors[i, ...] = base_colors[k, ...]
            i += 1
    return snp_class_colors


def snp_color_mapping_pc(points_with_labels):
    # points_with_labels: [num_points, 4]
    # points_with_colors: [num_points, 6]
    num_points = points_with_labels.size(0)
    device = points_with_labels.device
    assert points_with_labels.ndim == 2
    assert points_with_labels.size(-1) == 4
    snp_class_colors = ShapeNetPart_Colors() # color template, (50, 3)
    snp_class_colors = torch.Tensor(snp_class_colors).unsqueeze(0).to(device) # [1, 50, 3]
    points = points_with_labels[:, 0:3].unsqueeze(0) # [1, num_points, 3]
    labels = points_with_labels[:, 3].unsqueeze(0).long() # [1, num_points]
    colors = index_points(snp_class_colors, labels) # [1, num_points, 3]
    points_with_colors = torch.cat((points, colors), dim=-1).squeeze(0) # [num_points, 6]
    return points_with_colors


def snp_color_mapping_img(image_labeled):
    # image_labeled: [1, img_height, img_width]
    # image_colored: [3, img_height, img_width]
    device = image_labeled.device
    assert image_labeled.ndim==3 and image_labeled.size(0)==1
    img_height, img_width = image_labeled.size(1), image_labeled.size(2)
    num_pixels = int(img_height * img_width)
    snp_class_colors = ShapeNetPart_Colors() # (50, 3)
    background_color = np.array([255.0, 255.0, 255.0]).reshape(1, 3) # (1, 3)
    snp_class_colors_with_bg = np.concatenate((snp_class_colors, background_color), axis=0) # (51, 3)
    snp_class_colors_with_bg = torch.Tensor(snp_class_colors_with_bg).unsqueeze(0).to(device) # [1, 51, 3]
    image_labeled_reshaped = image_labeled.view(1, -1).long() # [1, num_pixels]
    indexed_colors = index_points(snp_class_colors_with_bg, image_labeled_reshaped) # [1, num_pixels, 3]
    image_colored = indexed_colors[0].permute(1, 0).contiguous().view(3, img_height, img_width) # [3, img_height, img_width]
    return image_colored


class ShapeNetPart_PC_Loader(torch.utils.data.Dataset):
    def __init__(self, root, mode):
        self.root = root
        self.mode = mode
        assert mode in ['train', 'test']
        self.class_file = os.path.join(root,'class_list.txt')
        self.list_file = os.path.join(root, mode + '_list.txt')
        self.class_list = [line.strip() for line in open(self.class_file, 'r')]  
        self.model_list = [line.strip() for line in open(self.list_file, 'r')]  
        self.num_models = len(self.model_list)
    def __getitem__(self, index):
        np.random.seed()
        model_name = self.model_list[index]
        class_name = model_name[:-5]
        class_id = self.class_list.index(class_name)
        model_file = os.path.join(self.root, 'point_cloud_labeled_2048', class_name, model_name + '.txt')
        pc = load_pc(model_file, ';') # (2048, 4)
        points = pc[:, 0:3] # (2048, 3)
        labels = pc[:, 3] # (2048,)
        if self.mode == 'train':
            points = random_anisotropic_scaling(points, 0.75, 1.25)
            points = random_translation(points, 0.10)
        return points, labels, class_id, model_name
    def __len__(self):
        return self.num_models
    
    
def ShapeNetPart_Balanced_SVI_Loader(root, batch_size, running_mode):
    np.random.seed()
    assert running_mode in ['train', 'test']
    num_classes = 16
    snp_classes = ['airplane', 'bag', 'cap', 'car', 'chair', 'earphone', 'guitar', 'knife', 'lamp', 'laptop', 
    'motorbike', 'mug', 'pistol', 'rocket', 'skateboard', 'table']
    train_class_index_dict = {'airplane': [1, 2349], 'bag': [1, 62], 'cap': [1, 44], 'car': [1, 740], 'chair': [1, 3054],
    'earphone': [1, 55], 'guitar': [1, 628], 'knife': [1, 312], 'lamp': [1, 1261], 'laptop': [1, 368], 'motorbike': [1, 151],
    'mug': [1, 146], 'pistol': [1, 239], 'rocket': [1, 54], 'skateboard': [1, 121], 'table': [1, 4423]}
    test_class_index_dict = {'airplane': [2350, 2690], 'bag': [63, 76], 'cap': [45, 55], 'car': [741, 898], 'chair': [3055, 3758], 
    'earphone': [56, 69], 'guitar': [629, 787], 'knife': [313, 392], 'lamp': [1262, 1547], 'laptop': [369, 451], 'motorbike': [152, 202],
    'mug': [147, 184], 'pistol': [240, 283], 'rocket': [55, 66], 'skateboard': [122, 152], 'table': [4424, 5271]}
    total_class_index_dict = {'airplane': [1, 2690], 'bag': [1, 76], 'cap': [1, 55], 'car': [1, 898], 'chair': [1, 3758], 
    'earphone': [1, 69], 'guitar': [1, 787], 'knife': [1, 392], 'lamp': [1, 1547], 'laptop': [1, 451], 'motorbike': [1, 202], 
    'mug': [1, 184], 'pistol': [1, 283], 'rocket': [1, 66], 'skateboard': [1, 152], 'table': [1, 5271]}
    if running_mode == 'train':
        target_class_index_dict = train_class_index_dict
    if running_mode == 'test':
        target_class_index_dict = test_class_index_dict
    el = [1, 2]
    az = [1, 2, 3, 4, 5, 6, 7, 8]
    suffixes = []
    for el_id in el:
        for az_id in az:
            suffixes.append(str(el_id) + '_' + str(az_id))
    vi = []
    vl = []
    class_id = []
    model_name_with_suff = []
    to_tensor = transforms.Compose([transforms.ToTensor()])
    selected_classes = np.random.choice(num_classes, batch_size)
    for bid in range(batch_size):
        class_id_this = selected_classes[bid]
        class_id.append(class_id_this)
        class_name = snp_classes[class_id_this]
        start_model_index = target_class_index_dict[class_name][0]
        end_model_index = target_class_index_dict[class_name][1]
        selected_model_index = np.random.randint(low=start_model_index, high=end_model_index+1)
        selected_model_suffix = np.random.choice(suffixes)
        model_name_with_suff_this = class_name + '_' + align_number(selected_model_index, 4) + '_' + selected_model_suffix
        model_name_with_suff.append(model_name_with_suff_this)
        view_image = Image.open(os.path.join(root, 'mv_image_16', class_name, model_name_with_suff_this + '.jpg'))
        view_image = to_tensor(view_image).unsqueeze(0) # [1, 3, img_size, img_size]
        view_label = Image.open(os.path.join(root, 'mv_image_labeled_16', class_name, model_name_with_suff_this + '.png'))
        view_label = torch.from_numpy(np.array(view_label)).unsqueeze(0).unsqueeze(0) # [1, 1, img_size, img_size]
        vi.append(view_image)
        vl.append(view_label)
    vi = torch.cat(vi, dim=0) # [batch_size, 3, img_size, img_size]
    vl = torch.cat(vl, dim=0) # [batch_size, 3, img_size, img_size]
    class_id = torch.from_numpy(np.array(class_id))
    return vi, vl, class_id, model_name_with_suff


class ShapeNetPart_Joint_Loader(torch.utils.data.Dataset):
    def __init__(self, data_root, expt_root, mode):
        self.data_root = data_root
        self.expt_root = expt_root
        self.mode = mode
        assert mode in ['train', 'test']
        self.class_file = os.path.join(data_root,'class_list.txt')
        self.list_file = os.path.join(data_root, mode + '_list.txt')
        self.class_list = [line.strip() for line in open(self.class_file, 'r')]  
        self.model_list = [line.strip() for line in open(self.list_file, 'r')]
        self.num_models = len(self.model_list)
    def __getitem__(self, index):
        np.random.seed()
        model_name = self.model_list[index]
        class_name = model_name[:-5]
        cid = self.class_list.index(class_name)
        # load point cloud
        pc_data = load_pc(os.path.join(self.data_root, 'point_cloud_labeled_2048', class_name, model_name + '.txt'), ';') # (2048, 4)
        points = pc_data[:, 0:3]
        labels = pc_data[:, 3]
        if self.mode == 'train':
            # load multi-view visibility
            mvv = load_pc(os.path.join(self.data_root, 'mv_visibility_2048x16', class_name, model_name + '_' + 'mvv' + '.xyz'), ' ')
            # load teacher knowledge 
            mv_cdw_t = load_pc(os.path.join(self.expt_root, 'mv_cdw', class_name, model_name + '_' + 'mv_cdw' + '.xyz'), ' ')
            # point cloud data augmentation
            points = random_anisotropic_scaling(points, 0.80, 1.20)
            points = random_translation(points, 0.10)
            return points, labels, mvv, cid, mv_cdw_t, model_name
        if self.mode == 'test':
            return points, labels, cid, model_name
    def __len__(self):
        return self.num_models


def ShapeNetCore_Balanced_SVI_Loader(root, batch_size):
    np.random.seed()
    num_classes = 55
    snc_classes = ['airplane', 'bag', 'basket', 'bathtub', 'bed', 'bench', 'birdhouse', 'bookshelf', 'bottle', 'bowl', 'bus', 
    'cabinet', 'camera', 'can', 'cap', 'car', 'cellphone', 'chair', 'clock', 'dishwasher', 'display', 'earphone', 'faucet', 
    'file_cabinet', 'flowerpot', 'guitar', 'helmet', 'jar', 'keyboard', 'knife', 'lamp', 'laptop', 'loudspeaker', 'mailbox', 
    'microphone', 'microwaves', 'motorbike', 'mug', 'piano', 'pillow', 'pistol', 'printer', 'remote', 'rifle', 'rocket', 
    'skateboard', 'sofa', 'stove', 'table', 'telephone', 'tower', 'train', 'trash_bin', 'washer', 'watercraft']
    total_class_index_dict = {'airplane': [1, 4045], 'bag': [1, 83], 'basket': [1, 113], 'bathtub': [1, 856], 'bed': [1, 233], 
    'bench': [1, 1813], 'birdhouse': [1, 73], 'bookshelf': [1, 452], 'bottle': [1, 498], 'bowl': [1, 186], 'bus': [1, 939], 
    'cabinet': [1, 1571], 'camera': [1, 113], 'can': [1, 108], 'cap': [1, 56], 'car': [1, 3514], 'cellphone': [1, 831], 
    'chair': [1, 6778], 'clock': [1, 651], 'dishwasher': [1, 93], 'display': [1, 1093], 'earphone': [1, 73], 'faucet': [1, 744], 
    'file_cabinet': [1, 298], 'flowerpot': [1, 602], 'guitar': [1, 797], 'helmet': [1, 162], 'jar': [1, 596], 'keyboard': [1, 65], 
    'knife': [1, 424], 'lamp': [1, 2318], 'laptop': [1, 460], 'loudspeaker': [1, 1597], 'mailbox': [1, 94], 'microphone': [1, 67],
    'microwaves': [1, 152], 'motorbike': [1, 337], 'mug': [1, 214], 'piano': [1, 239], 'pillow': [1, 96], 'pistol': [1, 307], 
    'printer': [1, 166], 'remote': [1, 66], 'rifle': [1, 2373], 'rocket': [1, 85], 'skateboard': [1, 152], 'sofa': [1, 3173], 
    'stove': [1, 218], 'table': [1, 8436], 'telephone': [1, 1089], 'tower': [1, 133], 'train': [1, 389], 'trash_bin': [1, 343], 
    'washer': [1, 169], 'watercraft': [1, 1939]}
    el = [1, 2]
    az = [1, 2, 3, 4, 5, 6, 7, 8]
    suffixes = []
    for el_id in el:
        for az_id in az:
            suffixes.append(str(el_id) + '_' + str(az_id))
    vi = []
    class_id = []
    model_name_with_suff = []
    to_tensor = transforms.Compose([transforms.ToTensor()])
    selected_classes = np.random.choice(num_classes, batch_size)
    for bid in range(batch_size):
        class_id_this = selected_classes[bid]
        class_id.append(class_id_this)
        class_name = snc_classes[class_id_this]
        start_model_index = total_class_index_dict[class_name][0]
        end_model_index = total_class_index_dict[class_name][1]
        selected_model_index = np.random.randint(low=start_model_index, high=end_model_index+1)
        selected_model_suffix = np.random.choice(suffixes)
        model_name_with_suff_this = class_name + '_' + align_number(selected_model_index, 4) + '_' + selected_model_suffix
        model_name_with_suff.append(model_name_with_suff_this)
        view_image = Image.open(os.path.join(root, 'mv_image_16', class_name, model_name_with_suff_this + '.jpg'))
        view_image = to_tensor(view_image).unsqueeze(0) # [1, 3, img_size, img_size]
        vi.append(view_image)
    vi = torch.cat(vi, dim=0) # [batch_size, 3, img_size, img_size]
    class_id = torch.from_numpy(np.array(class_id))
    return vi, class_id, model_name_with_suff


