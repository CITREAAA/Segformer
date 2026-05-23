本專案基於 SegFormer (b0) 架構進行微調（Fine-tuning），專為行動端（Flutter / Mobile ONNX Runtime）部署所打造的語義分割（Semantic Segmentation）專題。模型自帶自定義的 HWC 轉 CHW 維度調換、ImageNet 標準化預處理層，且輸出端已強制轉換為行動端友好的 int32 格式，實現零預處理、無痛對接手機端推論。

---

## 類別定義與色彩對照表 (10 Classes)
專案針對道路交通與避障場景，共定義了 10 個類別（9類 + 背景）。模型輸出的灰階像素值（Index）與對應的顏色、ID 如下表所示：

| Index | 類別名稱 (Label) | RGB 顏色 | Hex 色碼 | 說明 |
| :---: | :--- | :--- | :--- | :--- |
| **0** | background | (0, 0, 0) | #000000 | 背景 / 未分類區域 |
| **1** | Car | (89, 134, 179) | #5986B3 | 車輛 |
| **2** | crosswalk | (34, 117, 76) | #22754C | 斑馬線 |
| **3** | obstacle | (163, 73, 164) | #A349A4 | 障礙物 |
| **4** | person | (237, 28, 36) | #ED1C24 | 行人 |
| **5** | road | (61, 72, 204) | #3D48CC | 道路 / 柏油路面 |
| **6** | sidewalk | (255, 242, 0) | #FFF200 | 人行道 / 騎樓 |
| **7** | trafficlight_green | (61, 245, 61) | #3DF53D | 綠燈 |
| **8** | trafficlight_red | (255, 0, 204) | #FF00CC | 紅燈 |
| **9** | vegetation | (185, 122, 87) | #B97A57 | 植物 / 樹木花草 |

---

## 核心檔案功能說明

* train.py
  * 功能：模型訓練的主程式。支援凍結訓練（Freeze Train）與解凍微調，內部已更新 num_classes = 10。
  * 特色：特別針對小型資料集（如本專案的 230 張圖）進行了代數（Epoch）與 Batch Size 優化，避免過擬合（Overfitting）。支援斷點續訓功能（載入 last_epoch_weights.pth 即可接關）。

* segformer.py
  * 功能：模型的網路封裝。內含專為行動端打造的 SegFormerMobileWrapper 類別。
  * 特色：
    1. 封裝了 x.permute(0, 3, 1, 2)，讓手機端可以直接傳入常見的 [1, 512, 512, 3] (NHWC) 格式。
    2. 內建 ImageNet 標準化預處理（自動除以 255、減去均值並除以標準差），修正了台灣灰色磁磚人行道與馬路因數值相近而難以區分的問題。
    3. 輸出端使用 .to(torch.int32) 轉換，解決了行動端晶片及推論引擎對預設 int64 支援度差、導致記憶體錯位與大風吹色塊的問題。

* predict.py
  * 功能：電腦端推理與視覺化測試腳本。可載入 .pth 權重並指定 JPEGImages 內任意圖片進行語義分割，產出帶透明度顏色疊加的預覽圖（已修正 detect_image 的參數相容性）。

* export.py
  * 功能：ONNX 模型導出腳本。
  * 特色：啟用 simplify=True（ONNX-Simplifier），會自動優化並刪除冗餘節點。在 CPU 模式下（cuda=False）安全導出標準行動端 .onnx $int32 模型。

* remap.py
  * 功能：標籤像素值自動重映射（批次轉換）工具。
  * 特色：當多位同學標註規格不一致（例如舊格式 7 類與新格式 10 類排序錯亂）時，此工具能用程式碼一秒自動重映射所有舊 PNG 檔案的像素值，免去人工重新標註的痛苦。

* voc_annotation.py
  * 功能：資料集打散與分割腳本。會讀取混合後的所有圖片與標籤，重新將檔名寫入 VOC2007/ImageSets/Segmentation/train.txt 與 val.txt 中。

* predict.py
  * 測試指令：執行 python predict.py 後，輸入 VOCdevkit\VOC2007\JPEGImages\xxx.jpg 就能在電腦上即時看到疊加畫面。

* rename.py
  * 功能：從 CVAT 新下載的批次標註資料合併進原本的 VOCdevkit 大資料庫中。
  * 核心邏輯：
    1. 它會自動過濾並抓取帶有 _gtFine_labelIds.png 後綴的 CVAT 標籤檔。
    2. 自動將原圖（.jpg 或 .JPG）與標籤配對，並消除檔名雜訊。
    3. 統一格式改名：將它們批次重命名為標準格式 crosswalk_data_XXXX.jpg/.png。
    4. 修改欲新增圖片資料夾路徑 new_raw_img_dir, new_label_dir。
    5. 修改 merge_data(start_idx=xxx)，就能讓新資料從第 xxx 張開始往下接續。

---

## 核心操作流程 (Workflow)

### 步驟 1：資料集混合與標籤校正 (remap_labels.py)
1. 將 CVAT 中導出的最新標註（含有車子、紅綠燈號）檔案準備好。
2. 若有舊的 7 類標籤，請將其放入備份資料夾，並執行 python remap_labels.py 將其像素值格式重對齊為全新的 10 類格式。
3. 把新、舊標籤與原圖全部混入 VOCdevkit/VOC2007/ 的對應資料夾。

### 步驟 2：重新生成資料集清單 (voc_annotation.py)
在專案根目錄下執行：
python voc_annotation.py
這會自動重新掃描所有圖片，並打散為訓練集與驗證集。

### 步驟 3：啟動深度學習訓練 (train.py)
確認 train.py 中的 num_classes = 10 後執行：
python train.py
* 接關提示：如遇到顯示卡記憶體不足斷掉，可將 Unfreeze_batch_size 降為 2，並將 model_path 改為 logs/ 裡最新的斷點權重檔（例如 last_epoch_weights.pth），修改 Init_Epoch 後即可繼續訓練。

### 步驟 4：模型導出行動端 ONNX (export_now.py)
訓練完成後，拿著 logs/best_epoch_weights.pth 執行轉檔腳本：
python export_now.py
成功後會看到 ONNX simplified successfully. Export Complete. 提示，並產出 segformer_new.onnx。