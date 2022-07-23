#
# Authors: Wei-Hong Li

import os
import copy
import torch
import torch.nn.functional as F

from torchvision import transforms
from torch.utils.data import DataLoader

def get_model(args, tasks_outputs):
	# Return multi-task learning model or single-task model
	if args.backbone == 'segnet':
		from models.segnet import SegNet
		backbone = SegNet()
		backbone_channels = 64
	# TODO: more backbones

	if args.method == 'single-task':
		from models.models import SingleTaskModel
		task = args.task
		head = get_head(args.head, backbone_channels, tasks_outputs[task])
		model = SingleTaskModel(backbone, head, task)
	elif args.method == 'vanilla':
		selected_tasks_outputs = {}
		for task, task_output in tasks_outputs.items():
			if task in args.tasks:
				selected_tasks_outputs[task] = task_output
		from models.models import MultiTaskModel
		heads = torch.nn.ModuleDict({task: get_head(args.head, backbone_channels, task_output) for task, task_output in zip(args.tasks, selected_tasks_outputs.values())})
		model = MultiTaskModel(backbone, heads, args.tasks)

	return model

def get_stl_model(args, tasks_outputs, task):
	# Return single-task learning models
	backbone_name = args.backbone
	if backbone_name == 'segnet':
		from models.segnet import SegNet
		backbone = SegNet()
		backbone_channels = 64
	from models.models import SingleTaskModel
	head = get_head(args.head, backbone_channels, tasks_outputs[task])
	model = SingleTaskModel(backbone, head, task)
	return model

def get_head(head, backbone_channels, task_output):
	""" Return the decoder head """
	if head == 'segnet_head':
		from models.segnet import SegNet_head
		return SegNet_head(backbone_channels, task_output)


