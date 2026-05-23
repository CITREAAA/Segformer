import os
import shutil
from tqdm import tqdm

# --- 1. 設定路徑 ---
new_raw_img_dir = r"C:\Users\user\桌面\專題\video4_label\origin"
new_label_dir = r"C:\Users\user\桌面\專題\video4_label\label"

target_img_dir = "VOCdevkit/VOC2007/JPEGImages"
target_label_dir = "VOCdevkit/VOC2007/SegmentationClass"

def merge_data(start_idx):
    # 取得新下載的所有標籤檔 (含 _gtFine_labelIds.png 的檔案)
    label_files = [f for f in os.listdir(new_label_dir) if "_gtFine_labelIds.png" in f]
    label_files.sort()
    
    print(f"找到 {len(label_files)} 張新標籤，從序號 {start_idx:04d} 開始接續...")
    
    count = 0
    for i, old_label_name in enumerate(tqdm(label_files)):
        new_idx = start_idx + i
        new_base_name = f"crosswalk_data_{new_idx:04d}"
        
        # A. 處理標籤檔 (.png)
        old_label_path = os.path.join(new_label_dir, old_label_name)
        new_label_path = os.path.join(target_label_dir, new_base_name + ".png")
        shutil.copy(old_label_path, new_label_path)
        
        # B. 處理原圖 (.jpg) - 改進後的邏輯
        # 先獲取不含 "_gtFine_labelIds.png" 的前綴部分
        # 例如: "crosswalk_data (1)_gtFine_labelIds.png" -> "crosswalk_data (1)"
        file_prefix = old_label_name.replace("_gtFine_labelIds.png", "")
        old_img_name = file_prefix + ".jpg"
        
        old_img_path = os.path.join(new_raw_img_dir, old_img_name)
        new_img_path = os.path.join(target_img_dir, new_base_name + ".jpg")
        
        if os.path.exists(old_img_path):
            shutil.copy(old_img_path, new_img_path)
            count += 1
        else:
            # 備用方案：如果還是找不到，嘗試檢查是否有 .JPG 大寫
            if os.path.exists(old_img_path.replace(".jpg", ".JPG")):
                shutil.copy(old_img_path.replace(".jpg", ".JPG"), new_img_path)
                count += 1
            else:
                print(f"\n[警告] 找不到對應原圖: {old_img_name}")

    print(f"\n處理完成！成功合併 {count} 組檔案至 VOCdevkit。")

if __name__ == "__main__":
    # 接續你目前的進度 (172張已存在，所以從 173 開始)
    merge_data(start_idx=215)