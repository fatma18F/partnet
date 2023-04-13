
import torch
import torch.nn as nn


def get_seg_loss (outputs, labels, m3x3, m128x128, end_points, alpha = 0.0001): 
    criterion = torch.nn.NLLLoss()
    bs=outputs.size(0)
    #id3x3 = torch.eye(3, requires_grad=True).repeat(bs,1,1)
    #id128x128 = torch.eye(128, requires_grad=True).repeat(bs,1,1)
    #if outputs.is_cuda:
     #   id3x3=id3x3.cuda()
     #   id128x128=id128x128.cuda()
    #diff3x3 = id3x3-torch.bmm(m3x3,m3x3.transpose(1,2))
    #diff128x128 = id128x128-torch.bmm(m128x128,m128x128.transpose(1,2))
    logsoftmax = nn.LogSoftmax(dim=1)
    per_point_loss= criterion(logsoftmax(outputs), labels) #+ alpha * (torch.norm(diff3x3)+torch.norm(diff128x128)) / float(bs)
    end_points['per_point_seg_loss'] = per_point_loss
    per_shape_loss = torch.mean(per_point_loss, -1)
    end_points['per_shape_seg_loss'] = per_shape_loss
   # loss = torch.mean(per_shape_loss)
    return per_shape_loss, end_points 

from scipy.optimize import linear_sum_assignment
def hungarian_matching(pred_x, gt_x, curnmasks):
    """ pred_x, gt_x: B x nmask x n_point
        curnmasks: 
        return matcBhing_idx: B x nmask x 2 """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    curnmasks=torch.sum(curnmasks,dim=-1) 
    batch_size = gt_x.shape[0]
    nmask = gt_x.shape[1]
    matching_score = torch.matmul(gt_x, torch.transpose(pred_x, 2, 1 )) 
    sum=torch.unsqueeze(torch.sum(pred_x, dim=2), dim=1)+torch.sum(gt_x, 2, keepdim=True)-matching_score
    dim0,dim1,dim2=sum.shape
    matching_score=1-torch.divide(matching_score, torch.maximum(sum,torch.full((dim0,dim1,dim2), 1e-8).to(device)))
    matching_idx = torch.zeros((batch_size, nmask, 2),dtype =int)
    curnmasks = curnmasks.type(torch.int32)
    for i, curnmask in enumerate(curnmasks):
        matching_score_np = matching_score.clone().detach().cpu().numpy()       
        row_ind, col_ind = linear_sum_assignment(matching_score_np[i, :curnmask, :])
        matching_idx[i, :curnmask, 0] = torch.tensor(row_ind).to(device)
        matching_idx[i, :curnmask, 1] = torch.tensor(col_ind).to(device)  
    return matching_idx 
    
def iou(mask_pred, gt_x, gt_valid_pl, n_point, nmask, end_points):

     matching_idx=hungarian_matching(mask_pred,gt_x,gt_valid_pl)
     matching_idx.requires_grad = False
     end_points['matching_idx'] = matching_idx

     matching_idx_row = matching_idx[:, :, 0]
     idx=(matching_idx_row >= 0).nonzero(as_tuple=False)
     matching_idx_row=torch.cat( (torch.unsqueeze(idx[:, 0].int(),dim=-1) , torch.reshape( matching_idx_row, (-1, 1)) ),1)
     gt_x_matched = torch.reshape(gt_x[list(matching_idx_row.T.long())], (-1, nmask, n_point))
     
     matching_idx_column = matching_idx[:, :, 1]
     idx=(matching_idx_column >= 0).nonzero(as_tuple=False)
     matching_idx_column=torch.cat((torch.unsqueeze(idx[:, 0].int(),dim=-1 ) , torch.reshape( matching_idx_column, (-1, 1)) ),1)
     matching_idx_column_torch2=matching_idx_column.detach().cpu().numpy()
     pred_x_matched = torch.reshape(mask_pred[list(matching_idx_column.T)], (-1, nmask, n_point))
     
     matching_score = torch.sum(torch.multiply(gt_x_matched, pred_x_matched),2)
     iou_all = torch.div(matching_score, torch.sum(gt_x_matched, 2) + torch.sum(pred_x_matched, 2) - matching_score + 1e-8)
     end_points['per_shape_all_iou'] = iou_all
     meaniou = torch.div(torch.sum(torch.multiply(iou_all, gt_valid_pl), 1), torch.sum(gt_valid_pl, -1) + 1e-8) # B
     return meaniou, end_points



def get_ins_loss(mask_pred, gt_mask_pl, gt_valid_pl, end_points):

     """ Input:      mask_pred   B x K x N
                     mask_gt     B x K x N
                     gt_valid    B x K
     """
     gt_x=gt_mask_pl.float()
     gt_valid_pl=gt_valid_pl.float()
     _,num_ins , num_point= mask_pred.shape
     meaniou, end_points = iou( mask_pred, gt_x, gt_valid_pl, num_point, num_ins, end_points)
     print(meaniou)
     end_points['per_shape_mean_iou'] = meaniou
     loss = - torch.mean(meaniou)
     return loss, end_points