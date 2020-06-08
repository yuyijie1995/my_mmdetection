import PIL.Image as Image
import os
# 把多个图片拼接到一起对比
IMAGES_PATH_gt = './vised_imgs'  # 图片集地址
IMAGES_PATH_test = '/home/wrc/Competition/mmdetection-master/saved_images_submit_duck'  # 图片集地址
IMAGES_FORMAT = ['.png']  # 图片格式
IMAGE_SIZE = 1024  # 每张小图片的大小
IMAGE_ROW = 1  # 图片间隔，也就是合并成一张图后，一共有几行
IMAGE_COLUMN = 2  # 图片间隔，也就是合并成一张图后，一共有几列

IMAGE_SAVE_PATH = './save_com_imgs'  # 图片转换后的地址

# 获取图片集地址下的所有图片名称
image_names_gt = [name for name in os.listdir(IMAGES_PATH_gt) for item in IMAGES_FORMAT if
               os.path.splitext(name)[1] == item]
image_names_test = [name for name in os.listdir(IMAGES_PATH_test) for item in IMAGES_FORMAT if
               os.path.splitext(name)[1] == item]

# 定义图像拼接函数
def image_compose():
    to_image = Image.new('RGB', (IMAGE_COLUMN * IMAGE_SIZE, IMAGE_ROW * IMAGE_SIZE))  # 创建一个新图
    # 循环遍历，把每张图片按顺序粘贴到对应位置上
    for y in range(1, IMAGE_ROW + 1):
        for x in range(1, IMAGE_COLUMN + 1):
            from_image = Image.open(IMAGES_PATH_gt + image_names_gt[IMAGE_COLUMN * (y - 1) + x - 1]).resize(
                (IMAGE_SIZE, IMAGE_SIZE), Image.ANTIALIAS)
            to_image.paste(from_image, ((x - 1) * IMAGE_SIZE, (y - 1) * IMAGE_SIZE))
    return to_image.save(IMAGE_SAVE_PATH)  # 保存新图

def image_compose2(img_test,img_gt,imgname):
    save_image=Image.new('RGB', (IMAGE_COLUMN * IMAGE_SIZE, IMAGE_ROW * IMAGE_SIZE))  # 创建一个新图
    save_image.paste(img_gt, (0 * IMAGE_SIZE, 0 * IMAGE_SIZE))
    save_image.paste(img_test, (1 * IMAGE_SIZE, 0 * IMAGE_SIZE))
    save_image.save(IMAGE_SAVE_PATH+'/'+imgname,format='png')



for imgname in os.listdir(IMAGES_PATH_test):

    imgtest_path=os.path.join(IMAGES_PATH_test,imgname)
    imggt_path=os.path.join(IMAGES_PATH_gt,imgname)
    if os.path.isfile(imggt_path):
        img_gt=Image.open(imggt_path)
        img_test=Image.open(imgtest_path)
        image_compose2(img_test,img_gt,imgname)  # 调用函数
