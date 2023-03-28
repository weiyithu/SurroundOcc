import numpy as np
import torch
import chamfer

def voxel_to_vertices(voxel, img_metas, thresh=0.5):
    x = torch.linspace(0, voxel.shape[0] - 1, voxel.shape[0])
    y = torch.linspace(0, voxel.shape[1] - 1, voxel.shape[1])
    z = torch.linspace(0, voxel.shape[2] - 1, voxel.shape[2])
    X, Y, Z = torch.meshgrid(x, y, z)
    vv = torch.stack([X, Y, Z], dim=-1).to(voxel.device)

    vertices = vv[voxel > thresh]
    vertices[:, 0] = (vertices[:, 0] + 0.5) * (img_metas['pc_range'][3] - img_metas['pc_range'][0]) /  img_metas['occ_size'][0]  + img_metas['pc_range'][0]
    vertices[:, 1] = (vertices[:, 1] + 0.5) * (img_metas['pc_range'][4] - img_metas['pc_range'][1]) /  img_metas['occ_size'][1]  + img_metas['pc_range'][1]
    vertices[:, 2] = (vertices[:, 2] + 0.5) * (img_metas['pc_range'][5] - img_metas['pc_range'][2]) /  img_metas['occ_size'][2]  + img_metas['pc_range'][2]

    return vertices

def gt_to_vertices(gt, img_metas):
    gt[:, 0] = (gt[:, 0] + 0.5) * (img_metas['pc_range'][3] - img_metas['pc_range'][0]) /  img_metas['occ_size'][0]  + img_metas['pc_range'][0]
    gt[:, 1] = (gt[:, 1] + 0.5) * (img_metas['pc_range'][4] - img_metas['pc_range'][1]) /  img_metas['occ_size'][1]  + img_metas['pc_range'][1]
    gt[:, 2] = (gt[:, 2] + 0.5) * (img_metas['pc_range'][5] - img_metas['pc_range'][2]) /  img_metas['occ_size'][2]  + img_metas['pc_range'][2]
    return gt

def gt_to_voxel(gt, img_metas):
    voxel = np.zeros(img_metas['occ_size'])
    voxel[gt[:, 0].astype(np.int), gt[:, 1].astype(np.int), gt[:, 2].astype(np.int)] = gt[:, 3]

    return voxel

def eval_3d(verts_pred, verts_trgt, threshold=.5):
    d1, d2, idx1, idx2 = chamfer.forward(verts_pred.unsqueeze(0).type(torch.float), verts_trgt.unsqueeze(0).type(torch.float))
    dist1 = torch.sqrt(d1).cpu().numpy()
    dist2 = torch.sqrt(d2).cpu().numpy()
    cd = dist1.mean() + dist2.mean()
    precision = np.mean((dist1<threshold).astype('float'))
    recal = np.mean((dist2<threshold).astype('float'))
    fscore = 2 * precision * recal / (precision + recal)
    metrics = np.array([np.mean(dist1),np.mean(dist2),cd, precision,recal,fscore])
    return metrics

def evaluation_reconstruction(pred_occ, gt_occ, img_metas):
    results = []
    for i in range(pred_occ.shape[0]):
        
        vertices_pred = voxel_to_vertices(pred_occ[i], img_metas, thresh=0.25) #set low thresh for class imbalance problem
        vertices_gt = gt_to_vertices(gt_occ[i][..., :3], img_metas)
        
        metrics = eval_3d(vertices_pred.type(torch.double), vertices_gt.type(torch.double)) #must convert to double, a bug in chamfer
        results.append(metrics)
    return np.stack(results, axis=0)

def evaluation_semantic(pred_occ, gt_occ, img_metas, class_num):
    results = []

    for i in range(pred_occ.shape[0]):
        gt_i, pred_i = gt_occ[i].cpu().numpy(), pred_occ[i].cpu().numpy()
        gt_i = gt_to_voxel(gt_i, img_metas)
        mask = (gt_i != 255)
        score = np.zeros((class_num, 3))
        for j in range(class_num):
            if j == 0: #class 0 for geometry IoU
                score[j][0] += ((gt_i[mask] != 0) * (pred_i[mask] != 0)).sum()
                score[j][1] += (gt_i[mask] != 0).sum()
                score[j][2] += (pred_i[mask] != 0).sum()
            else:
                score[j][0] += ((gt_i[mask] == j) * (pred_i[mask] == j)).sum()
                score[j][1] += (gt_i[mask] == j).sum()
                score[j][2] += (pred_i[mask] == j).sum()

        results.append(score)
    return np.stack(results, axis=0)