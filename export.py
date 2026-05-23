from segformer import SegFormer_Segmentation

if __name__ == "__main__":
    # 這裡確保參數與你訓練時完全一致
    predictor = SegFormer_Segmentation(
        model_path = "logs/best_epoch_weights.pth", 
        num_classes = 10,
        phi = "b0",
        cuda = False 
    )

    # 執行轉檔
    predictor.convert_to_onnx(simplify=True, model_path="onnx/segformer.onnx")