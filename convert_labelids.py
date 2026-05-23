import os
import numpy as np
from PIL import Image
from tqdm import tqdm

# 這裡要根據你 labelIds 裡面的實際數值來對應
# 假設你的 labelIds 遵循 Cityscapes 標準：
# 或是直接將大於 0 的值對應到你的類別
# 如果你不確定原 ID 是多少，這個腳本會先印出來給你看
seg_dir = "VOCdevkit/VOC2007/SegmentationClass/"

def convert_ids():
    files = [f for f in os.listdir(seg_dir) if f.endswith(".png")]
    print("正在檢查並轉換 Label IDs...")
    
    for filename in tqdm(files):
        path = os.path.join(seg_dir, filename)
        img = Image.open(path)
        img_array = np.array(img)
        
        # 取得這張圖裡面所有的像素值
        unique_ids = np.unique(img_array)
        
        # 如果你發現雖然有標註，但 Key 只有 0，
        # 可能是因為之前的 RGB 腳本把圖刷壞了。
        # 請確保你是拿「原始改名後」的 labelIds.png 來跑這個。
        
        # 範例轉換邏輯（請根據實際 ID 修改）：
        # new_array = np.zeros_like(img_array)
        # new_array[img_array == 7] = 1  # 假設 7 是路
        # ... 以此類推
        
        # 如果你的圖目前是全黑(只有0)，請重新從原始標註包解壓 labelIds 覆蓋回來
        
    print("檢查完成。")

if __name__ == "__main__":
    convert_ids()