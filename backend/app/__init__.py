!pip install tensorflow gradio opencv-python pillow numpy matplotlib transformers ultralytics scikit-learn seaborn torch segmentation-models-pytorch grad-cam shap --quiet
!pip install requests basicsr gfpgan facexlib -q
!pip install realesrgan -q

import os
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image, ImageEnhance
import gradio as gr
import random
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, precision_recall_curve, auc, average_precision_score
from sklearn.utils import class_weight
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import KFold
from transformers import ViTImageProcessor, ViTForImageClassification
from ultralytics import YOLO
import torch
import torchvision.models as models
from torchvision import transforms
from pytorch_grad_cam import ScoreCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import segmentation_models_pytorch as smp
from datetime import datetime
import logging
from io import BytesIO
from torchvision.ops import nms
from scipy.ndimage import gaussian_filter
import pickle
import requests
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from torch.utils.data import DataLoader

logging.getLogger("transformers").setLevel(logging.ERROR)

# ---------------- CONFIG ----------------
IMAGE_SIZE = 224
classes = ['oral_cancer', 'healthy']
image_data, label_data = [], []
BATCH_SIZE = 8
EPOCHS = 5
CONFIDENCE_THRESHOLD = 0.7
SEGMENTATION_THRESHOLD = 0.5
USE_VIT = True
MODEL_PATH = 'best_model.pt'
CNN_MODEL_PATH = 'best_cnn_model.pt'
DATA_PATH = 'data.pkl'
PATCH_SIZE = IMAGE_SIZE // 16  # For ViT-base-patch16, 224/16=14

# Load saved training data if exists
if os.path.exists(DATA_PATH):
    with open(DATA_PATH, 'rb') as f:
        image_data, label_data = pickle.load(f)
    print(f"Loaded {len(image_data)} saved training images.")

# ---------------- MODELS ----------------
yolo_model = YOLO("yolov8l.pt")
feature_extractor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
unet_model = smp.Unet(
    encoder_name="resnet50",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1
).eval()
pretrained_path = "unet_pretrained.pth"
use_segmentation = False
if os.path.exists(pretrained_path):
    try:
        unet_model.load_state_dict(torch.load(pretrained_path, map_location=torch.device('cpu')))
        print(f" Loaded pretrained U-Net weights from {pretrained_path}")
        use_segmentation = True
    except Exception as e:
        print(f" Failed to load pretrained U-Net weights: {e}. Using default ImageNet-pretrained U-Net.")
        use_segmentation = True
else:
    print("U-Net weights not found. Using ImageNet-pretrained U-Net.")
    use_segmentation = True

pytorch_vit = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
pytorch_vit.classifier = torch.nn.Linear(768, len(classes))
if os.path.exists(MODEL_PATH):
    try:
        pytorch_vit.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
        print(f" Loaded pretrained PyTorch ViT model from {MODEL_PATH}")
    except Exception as e:
        print(f" Failed to load pretrained PyTorch ViT model: {e}")
pytorch_vit.eval()

cnn_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
cnn_model = models.resnet50(weights='IMAGENET1K_V1')
cnn_model.fc = torch.nn.Linear(cnn_model.fc.in_features, len(classes))
if os.path.exists(CNN_MODEL_PATH):
    try:
        cnn_model.load_state_dict(torch.load(CNN_MODEL_PATH, map_location=torch.device('cpu')))
        print(f" Loaded pretrained CNN model from {CNN_MODEL_PATH}")
    except Exception as e:
        print(f" Failed to load pretrained CNN model: {e}")
cnn_model.eval()

# ---------------- IMAGE PREPROCESSING ----------------
def apply_clahe(image_np):
    try:
        lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        merged = cv2.merge((l, a, b))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
    except Exception as e:
        print(f" CLAHE failed: {e}")
        return image_np

def augment(image):
    try:
        image = Image.fromarray(np.uint8(image))
        image = ImageEnhance.Brightness(image).enhance(random.uniform(0.8, 1.2))
        image = ImageEnhance.Contrast(image).enhance(random.uniform(0.8, 1.2))
        image = ImageEnhance.Sharpness(image).enhance(random.uniform(0.8, 1.2))
        if random.random() > 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() > 0.5:
            image = image.rotate(random.uniform(-15, 15))
        if random.random() > 0.5:
            image = ImageEnhance.Color(image).enhance(random.uniform(0.8, 1.2))
        return np.array(image)
    except Exception as e:
        print(f"Augmentation failed: {e}")
        return image

def preprocess_image(image_np, return_pytorch=False):
    try:
        if return_pytorch:
            img = cv2.resize(image_np, (IMAGE_SIZE, IMAGE_SIZE))
            img = img.transpose(2, 0, 1) / 255.0
            return torch.tensor(img, dtype=torch.float32).unsqueeze(0)
        return feature_extractor(images=image_np, return_tensors="tf")['pixel_values']
    except Exception as e:
        print(f" Preprocessing failed: {e}")
        return None

def preprocess_image_tf(image):
    try:
        image = tf.cast(image, tf.float32)
        image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
        image = image / 255.0
        return feature_extractor(images=image.numpy(), return_tensors="tf")['pixel_values']
    except Exception as e:
        print(f"TF preprocessing failed: {e}")
        return None

# ---------------- OBJECT DETECTION ----------------
def compute_iou(box1, box2):
    try:
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - inter_area
        return inter_area / union_area if union_area > 0 else 0
    except Exception as e:
        print(f" IoU computation failed: {e}")
        return 0

def detect_objects(image_np):
    try:
        if image_np.dtype != np.uint8:
            image_np = image_np.astype(np.uint8)
        if image_np.shape[-1] != 3:
            raise ValueError(f"Expected RGB image with shape (H, W, 3), got {image_np.shape}")

        detections = []
        mask = segment_affected_area(image_np)
        if np.sum(mask) > 0:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:3]
                for contour in sorted_contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    x1 = max(0, x - 15)
                    y1 = max(0, y - 15)
                    x2 = min(image_np.shape[1], x + w + 15)
                    y2 = min(image_np.shape[0], y + h + 15)
                    detections.append(((x1, y1, x2, y2), 0.95))
                print(f" Found {len(detections)} segmentation-based detections.")

        result = yolo_model(image_np, conf=0.5)[0]
        boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else np.array([])
        scores = result.boxes.conf.cpu().numpy() if result.boxes is not None else np.array([])

        if len(boxes) > 0:
            valid_boxes = []
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box)
                yolo_box = (x1, y1, x2, y2)
                box_mask = np.zeros_like(mask)
                cv2.rectangle(box_mask, (x1, y1), (x2, y2), 255, -1)
                intersection = np.logical_and(mask, box_mask).sum()
                box_area = (x2 - x1) * (y2 - y1)
                mask_area = np.sum(mask > 0)
                iou_mask = intersection / (box_area + mask_area - intersection + 1e-8)
                if iou_mask > 0.3:
                    valid_boxes.append((i, scores[i]))

            if valid_boxes:
                indices, valid_scores = zip(*valid_boxes)
                boxes = boxes[list(indices)]
                scores = np.array(valid_scores)
                nms_indices = nms(torch.tensor(boxes, dtype=torch.float32),
                                  torch.tensor(scores, dtype=torch.float32),
                                  iou_threshold=0.45).numpy()
                dynamic_threshold = max(0.4, np.percentile(scores, 80) if len(scores) > 0 else 0.4)
                for idx in nms_indices:
                    if scores[idx] > dynamic_threshold:
                        box = boxes[idx]
                        x1, y1, x2, y2 = map(int, box)
                        x1 = max(0, x1 - 10)
                        y1 = max(0, y1 - 10)
                        x2 = min(image_np.shape[1], x2 + 10)
                        y2 = min(image_np.shape[0], y2 + 10)
                        detections.append(((x1, y1, x2, y2), float(scores[idx])))
                print(f"Found {len(detections)} YOLO detections after filtering.")

        detections = sorted(detections, key=lambda x: x[1], reverse=True)[:3]
        return detections
    except Exception as e:
        print(f"Object detection failed: {e}")
        return []

# ---------------- RESHAPE TRANSFORM FOR ViT ----------------
def reshape_transform_vit(tensor):
    tensor = tensor[:, 1:, :]  # Remove CLS token
    tensor = tensor.view(tensor.size(0), PATCH_SIZE, PATCH_SIZE, tensor.size(2))
    tensor = tensor.transpose(2, 3).transpose(1, 2)  # To (bs, c, h, w)
    return tensor

# ---------------- SCORE-CAM (Latest Explainability) ----------------
def generate_score_cam(image_np, pred_idx=None, model=pytorch_vit, is_vit=True):
    try:
        if image_np.dtype != np.uint8:
            image_np = image_np.astype(np.uint8)
        if image_np.shape[-1] != 3:
            raise ValueError(f"Expected RGB image with shape (H, W, 3), got {image_np.shape}")

        x1, y1, x2, y2 = 0, 0, image_np.shape[1], image_np.shape[0]
        region = image_np.copy()

        if is_vit:
            input_tensor = preprocess_image(region, return_pytorch=True)
            reshape_func = reshape_transform_vit
            target_layers = [model.vit.encoder.layer[-1].attention]
        else:
            img_pil = Image.fromarray(region)
            input_tensor = cnn_transform(img_pil).unsqueeze(0)
            reshape_func = None
            target_layers = [model.layer4[-1]]
        if input_tensor is None:
            raise ValueError("Image preprocessing failed.")

        model.eval()
        cam = ScoreCAM(model=model, target_layers=target_layers, reshape_transform=reshape_func)

        if pred_idx is None:
            with torch.no_grad():
                if is_vit:
                    outputs = model(input_tensor)
                    logits = outputs.logits
                else:
                    logits = model(input_tensor)
                pred_idx = torch.argmax(logits, dim=-1).item()
        targets = [ClassifierOutputTarget(pred_idx)]

        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
        print(f"Score-CAM raw min: {np.min(grayscale_cam):.4f}, max: {np.max(grayscale_cam):.4f}")

        cam_min = np.min(grayscale_cam)
        cam_max = np.max(grayscale_cam)
        if cam_max - cam_min > 1e-8:
            normalized_cam = (grayscale_cam - cam_min) / (cam_max - cam_min)
        else:
            normalized_cam = np.ones_like(grayscale_cam) * 0.5
            print(" Score-CAM values uniform, using fallback normalization.")

        normalized_cam = cv2.resize(normalized_cam, (region.shape[1], region.shape[0]), interpolation=cv2.INTER_LINEAR)
        heatmap = cv2.applyColorMap(np.uint8(255 * normalized_cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(region, 0.5, heatmap, 0.7, 0)
        output_image = image_np.copy()
        output_image[y1:y2, x1:x2] = overlay
        return Image.fromarray(output_image)
    except Exception as e:
        print(f" Score-CAM generation failed: {e}")
        return Image.fromarray(image_np)

# ---------------- ESRGAN INTEGRATION ----------------
def setup_esrgan():
    model_name = 'RealESRGAN_x4plus'
    model_path = os.path.join('weights', f'{model_name}.pth')
    os.makedirs('weights', exist_ok=True)
    if not os.path.exists(model_path):
        print(f" Downloading {model_name} model...")
        url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5/RealESRGAN_x4plus.pth'
        with open(model_path, 'wb') as f:
            f.write(requests.get(url).content)
    model = RRDBNet(
        num_in_ch=3, num_out_ch=3,
        num_feat=64, num_block=23,
        num_grow_ch=32, scale=4
    )
    upsampler = RealESRGANer(
        scale=4,
        model_path=model_path,
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=torch.cuda.is_available()
    )
    return upsampler

upsampler = setup_esrgan()

def super_resolve(img_pil):
    try:
        img_np = np.array(img_pil)[:, :, ::-1]
        output, _ = upsampler.enhance(img_np, outscale=4)
        output_rgb = output[:, :, ::-1]
        return Image.fromarray(output_rgb)
    except Exception as e:
        print(f" Error during super-resolution: {e}")
        return Image.fromarray(np.array(img_pil))

# ---------------- IMAGE PROCESSING ----------------
def draw_boxes_and_mask(image_np, detections, mask):
    try:
        if image_np.dtype != np.uint8:
            image_np = image_np.astype(np.uint8)
        for box, score in detections:
            x1, y1, x2, y2 = box
            color = (0, 255, 0) if score < 0.7 else (255, 0, 0)
            cv2.rectangle(image_np, (x1, y1), (x2, y2), color, 3)
            cv2.putText(image_np, f"{score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        return Image.fromarray(image_np)
    except Exception as e:
        print(f"Draw boxes and mask failed: {e}")
        return Image.fromarray(image_np)

def segment_affected_area(image_np):
    try:
        if not use_segmentation:
            return np.zeros((image_np.shape[0], image_np.shape[1]), dtype=np.uint8)
        img = cv2.resize(image_np, (IMAGE_SIZE, IMAGE_SIZE))
        img_tensor = torch.tensor(img.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0) / 255.0
        with torch.no_grad():
            mask = unet_model(img_tensor).squeeze().numpy()
        mask = (mask > SEGMENTATION_THRESHOLD).astype(np.uint8) * 255
        mask = cv2.resize(mask, (image_np.shape[1], image_np.shape[0]))
        return mask
    except Exception as e:
        print(f" Segmentation failed: {e}")
        return np.zeros((image_np.shape[0], image_np.shape[1]), dtype=np.uint8)

def get_segmented_image(image_np, mask):
    try:
        if image_np.dtype != np.uint8:
            image_np = image_np.astype(np.uint8)
        segmented = cv2.bitwise_and(image_np, image_np, mask=mask)
        return Image.fromarray(segmented)
    except Exception as e:
        print(f" Segmented image creation failed: {e}")
        return Image.fromarray(image_np)

# ---------------- UPLOAD FUNCTION ----------------
def upload_images(images, label):
    if not images:
        return " No images provided."
    if label not in classes:
        return f"Invalid label '{label}'. Choose from {classes}."
    uploaded_count = 0
    for img_path in images:
        try:
            img = Image.open(img_path).resize((IMAGE_SIZE, IMAGE_SIZE)).convert("RGB")
            arr = apply_clahe(np.array(img))
            arr = augment(arr)
            image_data.append(arr)
            label_data.append(classes.index(label))
            uploaded_count += 1
        except Exception as e:
            print(f" Failed to process image {img_path}: {e}")
    with open(DATA_PATH, 'wb') as f:
        pickle.dump((image_data, label_data), f)
    return f" Uploaded {uploaded_count} {label} images at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}. Total images: {len(image_data)}"

# ---------------- TRAINING ----------------
def train_model():
    global pytorch_vit, cnn_model, feature_extractor, classes, IMAGE_SIZE
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if not image_data or not label_data:
        return "No images uploaded.", None, None, None, None
    if len(image_data) < 20:
        return f" Insufficient data. Please upload at least 20 images total (current: {len(image_data)}).", None, None, None, None
    if len(set(label_data)) < len(classes):
        return "Upload at least one image per class.", None, None, None, None

    try:
        # Convert data to numpy arrays
        X = np.array(image_data)
        y = np.array(label_data, dtype=np.int64)
        print(f"Total samples: {len(X)}, Class distribution: {np.bincount(y)}")

        # Filter invalid labels
        valid_indices = np.isin(y, [0, 1])
        X = X[valid_indices]
        y = y[valid_indices]
        if len(X) < len(image_data):
            print(f"Filtered out {len(image_data) - len(X)} samples with invalid labels.")
        if len(X) < 20:
            return f" Insufficient valid data after filtering: {len(X)} samples.", None, None, None, None
        print(f"After filtering: Total samples: {len(X)}, Class distribution: {np.bincount(y)}")

        # Validate dataset
        for i, img in enumerate(X):
            if img.shape != (IMAGE_SIZE, IMAGE_SIZE, 3):
                return f"Invalid image shape at index {i}: {img.shape}. Expected ({IMAGE_SIZE}, {IMAGE_SIZE}, 3).", None, None, None, None

        # Enhanced data augmentation
        def enhanced_augment(image):
            try:
                image = Image.fromarray(np.uint8(image))
                image = ImageEnhance.Brightness(image).enhance(random.uniform(0.5, 1.5))
                image = ImageEnhance.Contrast(image).enhance(random.uniform(0.5, 1.5))
                image = ImageEnhance.Sharpness(image).enhance(random.uniform(0.5, 1.5))
                image = ImageEnhance.Color(image).enhance(random.uniform(0.5, 1.5))
                if random.random() > 0.5:
                    image = image.transpose(Image.FLIP_LEFT_RIGHT)
                if random.random() > 0.5:
                    image = image.transpose(Image.FLIP_TOP_BOTTOM)
                if random.random() > 0.5:
                    image = image.rotate(random.uniform(-60, 60))
                if random.random() > 0.5:
                    scale = random.uniform(0.6, 1.4)
                    w, h = image.size
                    new_w, new_h = int(w * scale), int(h * scale)
                    image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
                    left = (new_w - w) // 2
                    top = (new_h - h) // 2
                    image = image.crop((left, top, left + w, top + h))
                if random.random() > 0.5:
                    image_np = np.array(image)
                    noise = np.random.normal(0, 15, image_np.shape).astype(np.uint8)
                    image_np = np.clip(image_np + noise, 0, 255)
                    image = Image.fromarray(image_np)
                return np.array(image)
            except Exception as e:
                print(f" Enhanced augmentation failed: {e}")
                return image

        # Pre-augment dataset
        augmented_X = []
        augmented_y = []
        for img, lbl in zip(X, y):
            if lbl in [0, 1]:  # Ensure only valid labels are augmented
                augmented_X.append(img)
                augmented_y.append(lbl)
                for _ in range(6):
                    aug_img = enhanced_augment(img)
                    augmented_X.append(aug_img)
                    augmented_y.append(lbl)

        X = np.array(augmented_X)
        y = np.array(augmented_y, dtype=np.int64)
        print(f"After augmentation: Total samples: {len(X)}, Class distribution: {np.bincount(y)}")

        # K-fold cross-validation
        n_splits = 3
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        val_accuracies = []
        best_val_acc = 0.0
        best_cm = None
        best_report = None
        best_train_losses = None
        best_val_losses = None
        best_train_accs = None
        best_val_accs = None
        best_roc_data = None  # Now a dict
        best_pr_data = None   # Now a dict

        class ImageDataset(torch.utils.data.Dataset):
            def __init__(self, images, labels, processor):
                self.images = images
                self.labels = labels
                self.processor = processor

            def __len__(self):
                return len(self.images)

            def __getitem__(self, idx):
                img = self.images[idx]
                label = self.labels[idx]
                inputs = self.processor(images=img, return_tensors="pt")
                pixel_values = inputs['pixel_values'].squeeze(0)
                return pixel_values, label

        class ImageDatasetCNN(torch.utils.data.Dataset):
            def __init__(self, images, labels, transform):
                self.images = images
                self.labels = labels
                self.transform = transform

            def __len__(self):
                return len(self.images)

            def __getitem__(self, idx):
                img = self.images[idx]
                label = self.labels[idx]
                img = Image.fromarray(img)
                pixel_values = self.transform(img)
                return pixel_values, label

        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            print(f"Training fold {fold + 1}/{n_splits}...")
            train_idx = train_idx[train_idx < len(X)]
            val_idx = val_idx[val_idx < len(X)]
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            print(f"Fold {fold + 1}: Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
            print(f"Fold {fold + 1}: Unique training labels: {np.unique(y_train)}")

            # Create datasets and loaders
            train_dataset = ImageDataset(X_train, y_train, feature_extractor)
            val_dataset = ImageDataset(X_val, y_val, feature_extractor)
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
            train_dataset_cnn = ImageDatasetCNN(X_train, y_train, cnn_transform)
            val_dataset_cnn = ImageDatasetCNN(X_val, y_val, cnn_transform)
            train_loader_cnn = DataLoader(train_dataset_cnn, batch_size=BATCH_SIZE, shuffle=True)
            val_loader_cnn = DataLoader(val_dataset_cnn, batch_size=BATCH_SIZE, shuffle=False)

            # Prepare models
            for param in pytorch_vit.vit.parameters():
                param.requires_grad = False
            pytorch_vit.to(device)
            for param in cnn_model.parameters():
                param.requires_grad = False
            for param in cnn_model.fc.parameters():
                param.requires_grad = True
            cnn_model.to(device)

            # Compute class weights
            unique_labels = np.unique(y_train)
            if not np.all(np.isin(unique_labels, [0, 1])):
                return f"Invalid labels in training data: {unique_labels}. Expected [0, 1].", None, None, None, None
            weights = class_weight.compute_class_weight('balanced', classes=np.array([0, 1]), y=y_train)
            class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
            print(f"Fold {fold + 1}: Class weights: {class_weights}")

            # Loss and optimizers
            loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
            optimizer_vit = torch.optim.Adam(pytorch_vit.classifier.parameters(), lr=2e-5)
            scheduler_vit = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_vit, mode='min', factor=0.2, patience=3, min_lr=5e-7)
            optimizer_cnn = torch.optim.Adam(cnn_model.fc.parameters(), lr=2e-5)
            scheduler_cnn = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_cnn, mode='min', factor=0.2, patience=3, min_lr=5e-7)

            # Training loop
            train_losses = []
            val_losses = []
            train_accs = []
            val_accs = []
            patience = 5
            best_fold_val_loss = float('inf')
            counter = 0

            for epoch in range(EPOCHS):
                pytorch_vit.train()
                cnn_model.train()
                train_loss = 0.0
                train_correct = 0
                train_total = 0
                for batch_vit, batch_cnn in zip(train_loader, train_loader_cnn):
                    pixel_values_vit, labels = batch_vit
                    pixel_values_vit = pixel_values_vit.to(device)
                    labels = labels.to(device)
                    optimizer_vit.zero_grad()
                    outputs_vit = pytorch_vit(pixel_values_vit)
                    loss_vit = loss_fn(outputs_vit.logits, labels)
                    loss_vit.backward()
                    optimizer_vit.step()

                    pixel_values_cnn, _ = batch_cnn
                    pixel_values_cnn = pixel_values_cnn.to(device)
                    optimizer_cnn.zero_grad()
                    outputs_cnn = cnn_model(pixel_values_cnn)
                    loss_cnn = loss_fn(outputs_cnn, labels)
                    loss_cnn.backward()
                    optimizer_cnn.step()

                    train_loss += (loss_vit.item() + loss_cnn.item()) * labels.size(0) / 2
                    logits = (outputs_vit.logits + outputs_cnn) / 2
                    _, predicted = torch.max(logits, 1)
                    train_total += labels.size(0)
                    train_correct += (predicted == labels).sum().item()

                train_loss /= train_total
                train_acc = train_correct / train_total
                train_losses.append(train_loss)
                train_accs.append(train_acc)

                pytorch_vit.eval()
                cnn_model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0
                y_pred = []
                y_true = []
                y_scores_full = []  # Full probabilities for multi-class style
                with torch.no_grad():
                    for batch_vit, batch_cnn in zip(val_loader, val_loader_cnn):
                        pixel_values_vit, labels = batch_vit
                        pixel_values_vit = pixel_values_vit.to(device)
                        labels = labels.to(device)
                        outputs_vit = pytorch_vit(pixel_values_vit)
                        logits_vit = outputs_vit.logits

                        pixel_values_cnn, _ = batch_cnn
                        pixel_values_cnn = pixel_values_cnn.to(device)
                        logits_cnn = cnn_model(pixel_values_cnn)

                        logits = (logits_vit + logits_cnn) / 2
                        loss = loss_fn(logits, labels)
                        val_loss += loss.item() * labels.size(0)
                        _, predicted = torch.max(logits, 1)
                        val_total += labels.size(0)
                        val_correct += (predicted == labels).sum().item()
                        y_pred.extend(predicted.cpu().numpy())
                        y_true.extend(labels.cpu().numpy())
                        y_scores_full.extend(torch.softmax(logits, dim=-1).cpu().numpy())  # Full probs

                val_loss /= val_total
                val_acc = val_correct / val_total
                val_losses.append(val_loss)
                val_accs.append(val_acc)
                print(f"Epoch {epoch+1}/{EPOCHS}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

                scheduler_vit.step(val_loss)
                scheduler_cnn.step(val_loss)
                if val_loss < best_fold_val_loss:
                    best_fold_val_loss = val_loss
                    counter = 0
                else:
                    counter += 1
                    if counter >= patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break

            acc = val_acc
            val_accuracies.append(acc)
            print(f"Fold {fold + 1} validation accuracy: {acc:.2f}")

            # Convert to arrays
            y_true = np.array(y_true)
            y_scores_full = np.array(y_scores_full)
            y_pred = np.array(y_pred)

            # Binarize for multi-class style (even for binary)
            y_true_bin = label_binarize(y_true, classes=[0, 1])

            # ROC for each class
            fpr = {}
            tpr = {}
            roc_auc = {}
            for i in range(len(classes)):
                fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_scores_full[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])

            # PR for each class
            precision = {}
            recall = {}
            pr_auc = {}
            for i in range(len(classes)):
                precision[i], recall[i], _ = precision_recall_curve(y_true_bin[:, i], y_scores_full[:, i])
                pr_auc[i] = average_precision_score(y_true_bin[:, i], y_scores_full[:, i])

            if acc > best_val_acc:
                best_val_acc = acc
                used_ids = sorted(set(y_true) | set(y_pred))
                used_names = [classes[i] for i in used_ids]
                best_report = classification_report(
                    y_true, y_pred, labels=used_ids, target_names=used_names, output_dict=True
                )
                best_cm = confusion_matrix(y_true, y_pred, labels=used_ids)
                best_train_losses = train_losses
                best_val_losses = val_losses
                best_train_accs = train_accs
                best_val_accs = val_accs
                best_roc_data = {'fpr': fpr, 'tpr': tpr, 'roc_auc': roc_auc}
                best_pr_data = {'precision': precision, 'recall': recall, 'pr_auc': pr_auc}
                torch.save(pytorch_vit.state_dict(), MODEL_PATH)
                torch.save(cnn_model.state_dict(), CNN_MODEL_PATH)

        report_md = (
            "| Class | Precision | Recall | F1-Score | Support |\n"
            "|-------|-----------|--------|----------|---------|\n"
        )
        for name in used_names:
            r = best_report[name]
            report_md += (
                f"| {name} | {r['precision']:.2f} | {r['recall']:.2f} "
                f"| {r['f1-score']:.2f} | {int(r['support'])} |\n"
            )

        plt.figure(figsize=(6, 5))
        sns.heatmap(best_cm, annot=True, fmt="d", xticklabels=used_names, yticklabels=used_names)
        plt.title("Confusion Matrix (Best Fold)")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        buf = BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format="png")
        buf.seek(0)
        plt.close()
        cm_image = Image.open(buf).convert("RGB")

        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(best_train_accs, label='Train Accuracy')
        plt.plot(best_val_accs, label='Val Accuracy')
        plt.title('Model Accuracy (Best Fold)')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(best_train_losses, label='Train Loss')
        plt.plot(best_val_losses, label='Val Loss')
        plt.title('Model Loss (Best Fold)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        buf = BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format="png")
        buf.seek(0)
        plt.close()
        history_image = Image.open(buf).convert("RGB")

        # Plot ROC with multi-class style
        plt.style.use('dark_background')
        fpr, tpr, roc_auc = best_roc_data['fpr'], best_roc_data['tpr'], best_roc_data['roc_auc']
        plt.figure(figsize=(8, 6))
        colors = ['cyan', 'magenta']  # For 2 classes
        for i, color in enumerate(colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f'{classes[i]} (AUC = {roc_auc[i]:.2f})')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        buf = BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format="png")
        buf.seek(0)
        plt.close()
        roc_image = Image.open(buf).convert("RGB")

        # Plot PR with multi-class style
        plt.style.use('dark_background')
        precision_dict, recall_dict, pr_auc = best_pr_data['precision'], best_pr_data['recall'], best_pr_data['pr_auc']
        plt.figure(figsize=(8, 6))
        for i, color in enumerate(colors):
            plt.plot(recall_dict[i], precision_dict[i], color=color, lw=2, label=f'{classes[i]} (AP = {pr_auc[i]:.2f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        buf = BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format="png")
        buf.seek(0)
        plt.close()
        pr_image = Image.open(buf).convert("RGB")

        mean_val_acc = np.mean(val_accuracies)
        return (
            f" Training complete. Mean val acc across {n_splits} folds: {mean_val_acc:.2f}\n"
            f"Best fold val acc: {best_val_acc:.2f}\n\n### Classification Report (Best Fold)\n{report_md}",
            cm_image,
            history_image,
            roc_image,
            pr_image
        )
    except Exception as e:
        print(f" Training failed: {e}")
        return f" Training failed: {e}", None, None, None, None

# ---------------- PREDICTION ----------------
def predict(image):
    global pytorch_vit, cnn_model, feature_extractor, classes, IMAGE_SIZE, CONFIDENCE_THRESHOLD
    try:
        image_np = apply_clahe(np.array(Image.fromarray(image).resize((IMAGE_SIZE, IMAGE_SIZE)).convert("RGB")))
        detections = detect_objects(image_np)
        mask = segment_affected_area(image_np)
        segmented = get_segmented_image(image_np, mask)
        boxes_and_mask = draw_boxes_and_mask(image_np.copy(), detections, mask)
        enhanced = super_resolve(Image.fromarray(image_np))

        if detections:
            box, _ = detections[0]
            x1, y1, x2, y2 = box
            affected_area = image_np[y1:y2, x1:x2]
            if affected_area.size > 0:
                image_np_for_class = affected_area
            else:
                image_np_for_class = image_np
        else:
            image_np_for_class = image_np

        inputs_vit = preprocess_image(image_np_for_class, return_pytorch=True)
        if inputs_vit is None:
            raise ValueError("Image preprocessing failed.")

        img_pil = Image.fromarray(image_np_for_class)
        inputs_cnn = cnn_transform(img_pil).unsqueeze(0)

        pytorch_vit.eval()
        cnn_model.eval()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        inputs_vit = inputs_vit.to(device)
        inputs_cnn = inputs_cnn.to(device)
        pytorch_vit.to(device)
        cnn_model.to(device)

        with torch.no_grad():
            outputs_vit = pytorch_vit(inputs_vit)
            logits_vit = outputs_vit.logits
            probs_vit = torch.softmax(logits_vit, dim=-1).cpu().numpy()[0]
            logits_cnn = cnn_model(inputs_cnn)
            probs_cnn = torch.softmax(logits_cnn, dim=-1).cpu().numpy()[0]

        probs = (probs_vit + probs_cnn) / 2
        pred_idx = int(np.argmax(probs))
        pred_label = classes[pred_idx]

        label_dict = {classes[i]: float(probs[i]) for i in range(len(classes))}
        summary = f"Prediction: {pred_label} ({probs[pred_idx]*100:.2f}%)\n"
        if probs[pred_idx] < CONFIDENCE_THRESHOLD:
            summary += " Low confidence. Consult a specialist.\n"

        if pred_label == 'healthy':
            boxes_and_mask = Image.fromarray(image_np)
            score_cam = Image.fromarray(np.zeros_like(image_np))
        else:
            score_cam = generate_score_cam(image_np, pred_idx, model=cnn_model, is_vit=False)

        return label_dict, boxes_and_mask, segmented, score_cam, enhanced, summary
    except Exception as e:
        print(f" Prediction failed: {e}")
        return {}, Image.fromarray(image), Image.fromarray(image), Image.fromarray(image), Image.fromarray(image), f" Prediction failed: {e}"

# ---------------- GRADIO UI ----------------
upload_ui = gr.Interface(
    fn=upload_images,
    inputs=[gr.Files(file_count="multiple", label="Upload Images"), gr.Dropdown(choices=classes, label="Class Label")],
    outputs=gr.Textbox(label="Upload Status"),
    title="Upload Oral Cancer Data"
)
train_ui = gr.Interface(
    fn=train_model,
    inputs=[],
    outputs=[
        gr.Textbox(label="Status"),
        gr.Image(type="pil", label="Confusion Matrix"),
        gr.Image(type="pil", label="Training History"),
        gr.Image(type="pil", label="ROC Curve"),
        gr.Image(type="pil", label="Precision-Recall Curve")
    ],
    title="Train Oral Cancer Model"
)
predict_ui = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy", label="Upload Image"),
    outputs=[
        gr.Label(num_top_classes=2, label="Prediction Probabilities"),
        gr.Image(type="pil", label="Detected Regions + Segmentation"),
        gr.Image(type="pil", label="Segmented Area"),
        gr.Image(type="pil", label="Score-CAM Heatmap"),
        gr.Image(type="pil", label="Enhanced Image (Real-ESRGAN)"),
        gr.Textbox(label="Prediction Summary")
    ],
    title="Predict Oral Cancer"
)
esrgan_ui = gr.Interface(
    fn=super_resolve,
    inputs=gr.Image(type="pil", label="Upload Low-Quality Image"),
    outputs=gr.Image(type="pil", label="Enhanced Image"),
    title="Real-ESRGAN Super-Resolution",
    description="Enhance low-resolution images using Real-ESRGAN (4x upscaling)"
)

dashboard = gr.TabbedInterface(
    [upload_ui, train_ui, predict_ui, esrgan_ui],
    tab_names=["Upload", "Train", "Predict", "Enhance (REAL-ESRGAN)"],
    title="Oral Cancer Detection System",
    theme="soft"
)

if __name__ == "__main__":
    print(" Launching Oral Cancer Detection UI...")
    dashboard.launch()
