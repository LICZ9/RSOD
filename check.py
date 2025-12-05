import os

def count_images_in_folder(folder_path):
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']
    image_count = 0
    
    for filename in os.listdir(folder_path):
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            image_count += 1
    
    return image_count

# 使用示例
folder_path = '/sharefiles1/lichengzhoutest/projects/OursCornerLoss+Enhence/results/Newdata05ours'  # 替换为你的文件夹路径
print(f"文件夹中共有 {count_images_in_folder(folder_path)} 张图片")