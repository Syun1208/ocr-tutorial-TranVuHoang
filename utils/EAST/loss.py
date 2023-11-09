import torch
import torch.nn as nn
import pandas as pd
import os

ROOT = os.getcwd()
WORK_DIR = os.path.dirname(ROOT)
path_losses = os.path.join(ROOT, 'results')
os.makedirs(path_losses, exist_ok=True)


def get_dice_loss(gt_score, pred_score):
	inter = torch.sum(gt_score * pred_score)
	union = torch.sum(gt_score) + torch.sum(pred_score) + 1e-5
	return 1. - (2 * inter / union)
	 

def get_geo_loss(gt_geo, pred_geo):
	d1_gt, d2_gt, d3_gt, d4_gt, angle_gt = torch.split(gt_geo, 1, 1)
	d1_pred, d2_pred, d3_pred, d4_pred, angle_pred = torch.split(pred_geo, 1, 1)
	area_gt = (d1_gt + d2_gt) * (d3_gt + d4_gt)
	area_pred = (d1_pred + d2_pred) * (d3_pred + d4_pred)
	w_union = torch.min(d3_gt, d3_pred) + torch.min(d4_gt, d4_pred)
	h_union = torch.min(d1_gt, d1_pred) + torch.min(d2_gt, d2_pred)
	area_intersect = w_union * h_union
	area_union = area_gt + area_pred - area_intersect
	iou_loss_map = -torch.log((area_intersect + 1.0)/(area_union + 1.0))
	angle_loss_map = 1 - torch.cos(angle_pred - angle_gt)
	return iou_loss_map, angle_loss_map


class Loss(nn.Module):
	def __init__(self, weight_angle=10):
		super(Loss, self).__init__()
		self.weight_angle = weight_angle
		self.iou_losses = list()
		self.angle_losses = list()
		self.dice_losses = list()
	def forward(self, gt_score, pred_score, gt_geo, pred_geo, ignored_map):
		if torch.sum(gt_score) < 1:
			return torch.sum(pred_score + pred_geo) * 0
		
		classify_loss = get_dice_loss(gt_score, pred_score*(1-ignored_map))
		iou_loss_map, angle_loss_map = get_geo_loss(gt_geo, pred_geo)

		angle_loss = torch.sum(angle_loss_map*gt_score) / torch.sum(gt_score)
		iou_loss = torch.sum(iou_loss_map*gt_score) / torch.sum(gt_score)
		geo_loss = self.weight_angle * angle_loss + iou_loss
		self.iou_losses.append(iou_loss.detach().cpu().numpy().astype('float'))
		self.angle_losses.append(angle_loss.detach().cpu().numpy().astype('float'))
		self.dice_losses.append(classify_loss.detach().cpu().numpy().astype('float'))
		
		losses = pd.DataFrame({'iou_losses': self.iou_losses, 'angle_losses': self.angle_losses, 'dice_losses': self.dice_losses})
		losses.to_csv(os.path.join(path_losses, 'losses.csv'))

		print('Dice loss is {:.8f}, Angle loss is {:.8f}, IoU loss is {:.8f}'.format(classify_loss, angle_loss, iou_loss))
		return geo_loss + classify_loss
