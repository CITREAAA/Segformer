import numpy as np
from PIL import Image

# 1. 填入你同學標註好的任意一張 PNG 標籤路徑
# 請確保這是位於 SegmentationClass 資料夾內的 PNG 檔
mask_path = "VOCdevkit/VOC2007/SegmentationClass/crosswalk_data_0235.png" 

try:
    # 2. 以原始資料模式讀取（避免讀成 RGB 導致混淆）
    img = Image.open(mask_path)
    img_array = np.array(img)
    
    # 3. 找出這張圖片中所有出現過的像素值
    unique_pixels = np.unique(img_array)
    
    print("=" * 50)
    print(f"成功讀取標籤：{mask_path}")
    print(f"圖片的原始維度 (Shape): {img_array.shape}")
    print(f"圖片中實際包含的像素值 (Index): {unique_pixels}")
    print("=" * 50)
    print("\n💡 對照提示：")
    print("通常 0 是 background。如果這張圖有馬路，你應該會看到某個數字（例如 6 或 3）。")
    print("如果數字出現 255，代表邊緣被設定為忽略區域（VOC標準），這是正常的。")

except Exception as e:
    print(f"讀取失敗，請檢查檔案路徑。錯誤訊息: {e}")