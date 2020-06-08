# my_mmdetection
my_custom_mmdetection
记录一些比赛中使用的技巧和数据分析方法。
*******
### mmdetection的修改
#### 数据增强：
- [x] mixup
- [x] mosaic
- [x] 类似Stitcher中的mosaic，代码中标记为masaic
- [x] bboxjitter
- [x] gridmask(非训练版本)
- [x] Minus（减去模板的均值或序列图片均值）
#### 模型修改：
- [x] 新增bifpn实现
- [x] global roi 
- [ ] atss_Rcnn(代码可能有问题)
### data_make & data_analysis
- [x] json2voc and voc2coco
- [x] duck injucktion
- [x] make_gt_json
- [x] 反色数据
- [x] 训练验证集分割
### data_analysis
- [x] 可视化json
- [x] 可视化xml
- [x] 可视化每个类别的位置分布
- [x] 计算长宽比，大中小目标数量分布，各个类别数量分布
- [x] 把多个结果图片拼接起来对比
- [x] 多个结果文件的bbox打到一张图上和gt对比

