import torch
num_classes=10
# model_coco=torch.load('./pretrained/faster_rcnn_mdconv_c3-c5_group4_r50_fpn_1x_20190911-5591a7e4.pth')
model_coco=torch.load('./pretrained/cascade_rcnn_r50_fpn_dconv_c3-c5_1x_coco_20200130-2f1fca44.pth')
# model_coco['state_dict']['bbox_head.fc_cls.weight'].resize_(num_classes,1024)

# model_coco['state_dict']['bbox_head.fc_reg.weight'].resize_(num_classes*4,1024)
# model_coco['state_dict']['bbox_head.retina_cls.weight'].resize_(num_classes-1,256,3,3)
# model_coco['state_dict']['bbox_head.retina_cls.bias'].resize_(num_classes-1)
model_coco['state_dict']['roi_head.bbox_head.0.fc_cls.weight'].resize_(num_classes,1024)
model_coco['state_dict']['roi_head.bbox_head.1.fc_cls.weight'].resize_(num_classes,1024)
model_coco['state_dict']['roi_head.bbox_head.2.fc_cls.weight'].resize_(num_classes,1024)

model_coco['state_dict']['bbox_head.0.fc_cls.weight']=model_coco['state_dict']['roi_head.bbox_head.0.fc_cls.weight']
model_coco['state_dict']['bbox_head.1.fc_cls.weight']=model_coco['state_dict']['roi_head.bbox_head.1.fc_cls.weight']
model_coco['state_dict']['bbox_head.2.fc_cls.weight']=model_coco['state_dict']['roi_head.bbox_head.2.fc_cls.weight']


# model_coco['state_dict']['rpn_head.rpn_cls.weight']=model_coco['state_dict']['rpn_head.rpn_cls.weight'].repeat(3,1,1,1)
# model_coco['state_dict']['rpn_head.rpn_cls.weight']=model_coco['state_dict']['rpn_head.rpn_cls.weight'].resize_(5,256,1,1)
# model_coco['state_dict']['rpn_head.rpn_cls.bias']=model_coco['state_dict']['rpn_head.rpn_cls.bias'].repeat(3,)
# model_coco['state_dict']['rpn_head.rpn_cls.bias']=model_coco['state_dict']['rpn_head.rpn_cls.bias'].resize_(5,)
#
# model_coco['state_dict']['rpn_head.rpn_reg.weight']=model_coco['state_dict']['rpn_head.rpn_reg.weight'].repeat(3,1,1,1)
# model_coco['state_dict']['rpn_head.rpn_reg.weight']=model_coco['state_dict']['rpn_head.rpn_reg.weight'].resize_(20,256,1,1)
# model_coco['state_dict']['rpn_head.rpn_reg.bias']=model_coco['state_dict']['rpn_head.rpn_reg.bias'].repeat(3,)
# model_coco['state_dict']['rpn_head.rpn_reg.bias']=model_coco['state_dict']['rpn_head.rpn_reg.bias'].repeat(20,)

#
# model_coco['state_dict']['mask_head.0.conv_logits.weight'].resize_(num_classes,256,1,1)
# model_coco['state_dict']['mask_head.1.conv_logits.weight'].resize_(num_classes,256,1,1)
# model_coco['state_dict']['mask_head.2.conv_logits.weight'].resize_(num_classes,256,1,1)
#
# model_coco['state_dict']['bbox_head.fc_cls.bias'].resize_(num_classes)
# model_coco['state_dict']['bbox_head.fc_reg.bias'].resize_(num_classes*4)
#
# model_coco['state_dict']['bbox_head.retina_reg.weight'].resize_(num_classes)
# model_coco['state_dict']['bbox_head.retina_reg.bias'].resize_(num_classes*4)

model_coco['state_dict']['roi_head.bbox_head.0.fc_cls.bias'].resize_(num_classes)
model_coco['state_dict']['roi_head.bbox_head.1.fc_cls.bias'].resize_(num_classes)
model_coco['state_dict']['roi_head.bbox_head.2.fc_cls.bias'].resize_(num_classes)

model_coco['state_dict']['bbox_head.0.fc_cls.bias']=model_coco['state_dict']['roi_head.bbox_head.0.fc_cls.bias']
model_coco['state_dict']['bbox_head.1.fc_cls.bias']=model_coco['state_dict']['roi_head.bbox_head.1.fc_cls.bias']
model_coco['state_dict']['bbox_head.2.fc_cls.bias']=model_coco['state_dict']['roi_head.bbox_head.2.fc_cls.bias']

model_coco['state_dict']['bbox_head.1.fc_reg.weight']=model_coco['state_dict']['roi_head.bbox_head.1.fc_reg.weight']
model_coco['state_dict']['bbox_head.1.fc_reg.bias']=model_coco['state_dict']['roi_head.bbox_head.1.fc_reg.bias']
model_coco['state_dict']['bbox_head.2.fc_reg.weight']=model_coco['state_dict']['roi_head.bbox_head.2.fc_reg.weight']
model_coco['state_dict']['bbox_head.2.fc_reg.bias']=model_coco['state_dict']['roi_head.bbox_head.2.fc_reg.bias']



model_coco['state_dict']['bbox_head.0.fc_reg.weight']=model_coco['state_dict']['roi_head.bbox_head.0.fc_reg.weight']
model_coco['state_dict']['bbox_head.0.fc_reg.bias']=model_coco['state_dict']['roi_head.bbox_head.0.fc_reg.bias']

model_coco['state_dict']['bbox_head.0.shared_fcs.0.weight']=model_coco['state_dict']['roi_head.bbox_head.0.shared_fcs.0.weight']
model_coco['state_dict']['bbox_head.0.shared_fcs.0.bias']=model_coco['state_dict']['roi_head.bbox_head.0.shared_fcs.0.bias']
model_coco['state_dict']['bbox_head.0.shared_fcs.1.weight']=model_coco['state_dict']['roi_head.bbox_head.0.shared_fcs.1.weight']
model_coco['state_dict']['bbox_head.0.shared_fcs.1.bias']=model_coco['state_dict']['roi_head.bbox_head.0.shared_fcs.1.bias']

model_coco['state_dict']['bbox_head.1.shared_fcs.0.weight']=model_coco['state_dict']['roi_head.bbox_head.1.shared_fcs.0.weight']
model_coco['state_dict']['bbox_head.1.shared_fcs.0.bias']=model_coco['state_dict']['roi_head.bbox_head.1.shared_fcs.0.bias']
model_coco['state_dict']['bbox_head.1.shared_fcs.1.weight']=model_coco['state_dict']['roi_head.bbox_head.1.shared_fcs.1.weight']
model_coco['state_dict']['bbox_head.1.shared_fcs.1.bias']=model_coco['state_dict']['roi_head.bbox_head.1.shared_fcs.1.bias']

model_coco['state_dict']['bbox_head.2.shared_fcs.0.weight']=model_coco['state_dict']['roi_head.bbox_head.2.shared_fcs.0.weight']
model_coco['state_dict']['bbox_head.2.shared_fcs.0.bias']=model_coco['state_dict']['roi_head.bbox_head.2.shared_fcs.0.bias']
model_coco['state_dict']['bbox_head.2.shared_fcs.1.weight']=model_coco['state_dict']['roi_head.bbox_head.2.shared_fcs.1.weight']
model_coco['state_dict']['bbox_head.2.shared_fcs.1.bias']=model_coco['state_dict']['roi_head.bbox_head.2.shared_fcs.1.bias']

# model_coco['state_dict']['mask_head.0.conv_logits.bias'].resize_(num_classes)
# model_coco['state_dict']['mask_head.1.conv_logits.bias'].resize_(num_classes)
# model_coco['state_dict']['mask_head.2.conv_logits.bias'].resize_(num_classes)

torch.save(model_coco,"./pretrained/cascade_rcnn_r50_fpn_dconv_c3-c5_change_%s.pth"%num_classes)

