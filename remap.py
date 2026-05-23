import os
import numpy as np
from PIL import Image
from tqdm import tqdm

# ========================================================
# 設定路徑
# ========================================================
# 這裡填入你「最原始 7 類舊標籤」存放的資料夾
old_label_dir = "VOCdevkit/VOC2007/SegmentationClass_Old_Backup" 
# 轉換後要輸出到的資料夾（也就是 train.py 要讀取的資料夾）
new_label_dir = "VOCdevkit/VOC2007/SegmentationClass" 

os.makedirs(new_label_dir, exist_ok=True)

# ========================================================
# 最新對照表：根據你同學提供的新排序（0~9）
# 格式為 { 舊像素值 : 新像素值 }
# ========================================================
# 舊順序：0:bg, 1:crosswalk, 2:obstacle, 3:person, 4:road, 5:sidewalk, 6:vegetation
# 新順序：0:bg, 1:Car, 2:crosswalk, 3:obstacle, 4:person, 5:road, 6:sidewalk, 7:green, 8:red, 9:vegetation
label_mapping = {
    0: 0,  # background   -> background (0)
    1: 2,  # crosswalk    -> crosswalk (2)
    2: 3,  # obstacle     -> obstacle (3)
    3: 4,  # person       -> person (4)
    4: 5,  # road         -> road (5)
    5: 6,  # sidewalk     -> sidewalk (6)
    6: 9,  # vegetation   -> vegetation (9)
    255: 255 # VOC 邊緣忽略區域保持不變
}
# 註：舊標籤裡面本來就沒有獨立的 Car, trafficlight_green, trafficlight_red，
# 這些新類別會由你同學在 CVAT 標註的新圖中主動補上（佔用 index 1, 7, 8）。

# ========================================================
# 開始批次轉換
# ========================================================
print("開始將舊標籤像素值自動重映射至 10 類新規格...")
png_files = [f for f in os.listdir(old_label_dir) if f.endswith('.png')]

for filename in tqdm(png_files):
    old_path = os.path.join(old_label_dir, filename)
    new_path = os.path.join(new_label_dir, filename)
    
    img = Image.open(old_path)
    img_array = np.array(img)
    
    # 建立一個乾淨的矩陣來填寫新數值
    remapped_array = np.zeros_like(img_array)
    
    for old_val, new_val in label_mapping.items():
        remapped_array[img_array == old_val] = new_val
        
    # 轉為標準 8-bit 灰階
    new_img = Image.fromarray(remapped_array).convert("L") 
    new_img.save(new_path)

print(f"\n舊標籤轉換完成！已儲存至：{new_label_dir}")