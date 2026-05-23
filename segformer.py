import colorsys
import copy
import time
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from nets.segformer import SegFormer
from utils.utils import cvtColor, preprocess_input, resize_image, show_config

#---------------------------------------------------#
#   封裝類別：保險級別的行動端封裝
#---------------------------------------------------#
class SegFormerMobileWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        # 註冊 ImageNet 標準化參數
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        # 1. 自動維度檢查 (解決 HWC 還是 CHW 的問題)
        # 如果最後一維是 3 (代表是 Height, Width, Channel)
        if x.shape[-1] == 3:
            x = x.permute(0, 3, 1, 2) # 強制轉為 [Batch, Channel, Height, Width]

        # 2. 預處理：直接在模型內完成
        x = x.float() / 255.0
        x = (x - self.mean) / self.std
        
        # 3. 模型主體推論
        logits = self.model(x)
        if isinstance(logits, (list, tuple)):
            logits = logits[0]

        # 4. 後處理：放大到 512x512 並取最大機率類別
        x = F.interpolate(logits, size=(512, 512), mode='bilinear', align_corners=False)
        x = torch.argmax(x, dim=1)
        return x.to(torch.int32)

class SegFormer_Segmentation(object):
    _defaults = {
        "model_path"    : "logs/best_epoch_weights.pth",
        "num_classes"   : 10,
        "phi"           : "b0",
        "input_shape"   : [512, 512],
        "mix_type"      : 0,
        "cuda"          : True,
    }

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
        
        # 這裡建議手動寫死你的顏色表，確保 Python 與 Flutter 對齊
        self.colors = [
            (0, 0, 0),        # 0: background
            (89, 134, 179),   # 1: Car
            (34, 117, 76),    # 2: crosswalk
            (163, 73, 164),   # 3: obstacle
            (237, 28, 36),    # 4: person
            (61, 72, 204),    # 5: road
            (255, 242, 0),    # 6: sidewalk
            (61, 245, 61),    # 7: trafficlight_green
            (255, 0, 204),    # 8: trafficlight_red
            (185, 122, 87)    # 9: vegetation
        ]
        self.generate()

    def generate(self, onnx=False):
        self.net = SegFormer(num_classes=self.num_classes, phi=self.phi, pretrained=False)
        device = torch.device('cuda' if torch.cuda.is_available() and not onnx else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net = self.net.eval()
        if not onnx and self.cuda:
            self.net = nn.DataParallel(self.net).cuda()
        print(f"Weight {self.model_path} loaded successfully.")

    def convert_to_onnx(self, simplify, model_path):
        import onnx
        self.generate(onnx=True)
        
        # 使用封裝殼
        mobile_model = SegFormerMobileWrapper(self.net).eval().to('cpu')
        
        # 設定輸入 Shape 為 [1, 512, 512, 3] (迎合 Flutter 習慣的 HWC)
        dummy_input = torch.zeros(1, 512, 512, 3).to('cpu')
        
        print(f"Exporting ONNX model to {model_path}...")
        torch.onnx.export(
            mobile_model, 
            dummy_input, 
            model_path,
            verbose=False,
            opset_version=15, # 推薦版本 15
            do_constant_folding=True,
            input_names=["images"],
            output_names=["output_mask"],
            dynamic_axes=None
        )

        if simplify:
            try:
                import onnxsim
                model_onnx, check = onnxsim.simplify(onnx.load(model_path))
                if check: 
                    onnx.save(model_onnx, model_path)
                    print("ONNX simplified successfully.")
            except ImportError:
                print("onnx-simplifier not installed, skipping.")
        
        print("Export Complete.")

    def detect_image(self, image, count=False, name_classes=None):
        # 此處保持原本預覽邏輯，方便你在電腦端檢查模型
        image = cvtColor(image)
        old_img = copy.deepcopy(image)
        orininal_h, orininal_w = np.array(image).shape[0], np.array(image).shape[1]
        image_data, nw, nh = resize_image(image, (self.input_shape[1], self.input_shape[0]))
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda: images = images.cuda()
            pr = self.net(images)[0]
            pr = F.softmax(pr.permute(1,2,0), dim=-1).cpu().numpy()
            pr = pr[int((self.input_shape[0] - nh) // 2) : int((self.input_shape[0] - nh) // 2 + nh), \
                    int((self.input_shape[1] - nw) // 2) : int((self.input_shape[1] - nw) // 2 + nw)]
            pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation=cv2.INTER_LINEAR)
            pr = pr.argmax(axis=-1)
        
        seg_img = np.reshape(np.array(self.colors, np.uint8)[np.reshape(pr, [-1])], [orininal_h, orininal_w, -1])
        return Image.blend(old_img, Image.fromarray(np.uint8(seg_img)), 0.7)