import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# =====================================================
# REPRODUCIBILITY & DEVICE SETUP
# =====================================================

def set_random_seeds(seed=42):
    """Set random seeds for reproducible results"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_random_seeds(42)

def setup_device():
    """Setup device with proper error handling"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # Clear cache to avoid memory issues
        torch.cuda.empty_cache()
    else:
        device = torch.device("cpu")
        print("⚠️ Using CPU (GPU not available)")
    
    return device

device = setup_device()

# =====================================================
# CONFIGURATION
# =====================================================

class Config:
    """Simple, production-ready configuration"""
    # Dataset paths - UPDATE THESE
    REAL_PATH = "C:\\Users\\kg060\\Desktop\\projects\\deepfake-detection\\backend\\dataset\\real"
    FAKE_PATH = "C:\\Users\\kg060\\Desktop\\projects\\deepfake-detection\\backend\\dataset\\fake"
    
    # Model settings
    MODEL_PATH = "models/production_deepfake_detector.pth"
    IMG_SIZE = 224
    BATCH_SIZE = 16  # Conservative for stability
    NUM_EPOCHS = 30
    LEARNING_RATE = 0.0001
    DROPOUT_RATE = 0.3
    
    # Data loading
    NUM_WORKERS = 0 if os.name == 'nt' else 4  # No workers on Windows to avoid issues
    
    # Create directories
    os.makedirs("models", exist_ok=True)

print(f"📋 Configuration: Batch Size={Config.BATCH_SIZE}, Workers={Config.NUM_WORKERS}")

# =====================================================
# DATA PREPROCESSING
# =====================================================

# Training transforms with augmentation
train_transform = transforms.Compose([
    transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Validation transforms (no augmentation)
val_transform = transforms.Compose([
    transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# =====================================================
# DATASET CLASS
# =====================================================

class DeepfakeDataset(Dataset):
    """Clean dataset class with error handling"""
    
    def __init__(self, real_dir, fake_dir, transform=None):
        self.data = []
        self.transform = transform
        
        print("📂 Loading dataset...")
        
        # Load real images
        if os.path.exists(real_dir):
            real_files = [f for f in os.listdir(real_dir) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            for f in real_files:
                self.data.append((os.path.join(real_dir, f), 1))  # 1 = Real
            print(f"✅ Loaded {len(real_files)} REAL images")
        
        # Load fake images
        if os.path.exists(fake_dir):
            fake_files = [f for f in os.listdir(fake_dir) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            for f in fake_files:
                self.data.append((os.path.join(fake_dir, f), 0))  # 0 = Fake
            print(f"✅ Loaded {len(fake_files)} FAKE images")
        
        if len(self.data) == 0:
            raise ValueError("❌ No valid images found! Check your dataset paths.")
        
        # Shuffle for better training
        random.shuffle(self.data)
        print(f"📊 Total dataset size: {len(self.data)} images")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"⚠️ Error loading {img_path}: {e}")
            # Return zero tensor to prevent crashes
            return torch.zeros((3, Config.IMG_SIZE, Config.IMG_SIZE)), label

# =====================================================
# MODEL ARCHITECTURE
# =====================================================

class ProductionDeepfakeDetector(nn.Module):
    """Production-ready ResNet50 based detector"""
    
    def __init__(self, num_classes=2, dropout_rate=0.3):
        super(ProductionDeepfakeDetector, self).__init__()
        
        print("🏗️ Building ResNet50 model...")
        
        # Load pre-trained ResNet50
        self.resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        
        # Custom classifier head
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(256, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
        
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"✅ Model created: {total_params:,} total params, {trainable_params:,} trainable")
    
    def _initialize_weights(self):
        """Initialize weights for better training"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.resnet(x)

# =====================================================
# DATA LOADING
# =====================================================

def create_data_loaders():
    """Create optimized data loaders"""
    
    print("🔄 Creating data loaders...")
    
    # Create dataset
    full_dataset = DeepfakeDataset(
        Config.REAL_PATH, Config.FAKE_PATH, transform=train_transform
    )
    
    # Split dataset
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Apply validation transform to validation set
    val_dataset_copy = DeepfakeDataset(
        Config.REAL_PATH, Config.FAKE_PATH, transform=val_transform
    )
    val_indices = val_dataset.indices
    val_dataset = torch.utils.data.Subset(val_dataset_copy, val_indices)
    
    # Calculate class weights
    fake_count = sum(1 for _, label in full_dataset.data if label == 0)
    real_count = sum(1 for _, label in full_dataset.data if label == 1)
    
    if fake_count > 0 and real_count > 0:
        total = fake_count + real_count
        class_weights = torch.tensor([
            total / (2.0 * fake_count),
            total / (2.0 * real_count)
        ], dtype=torch.float32).to(device)
    else:
        class_weights = torch.tensor([1.0, 1.0], dtype=torch.float32).to(device)
    
    print(f"📊 Split: {train_size} train, {val_size} val")
    print(f"⚖️ Class weights: Fake={class_weights[0]:.3f}, Real={class_weights[1]:.3f}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader, class_weights

# =====================================================
# TRAINING FUNCTION
# =====================================================

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs):
    """Clean training function"""
    
    print(f"🚂 Starting training for {num_epochs} epochs...")
    
    best_val_acc = 0.0
    best_model_state = None
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(num_epochs):
        print(f"\n{'='*50}")
        print(f"EPOCH {epoch+1}/{num_epochs}")
        print('='*50)
        
        # ============ TRAINING PHASE ============
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        train_pbar = tqdm(train_loader, desc="🔥 Training")
        for inputs, labels in train_pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            
            # Update progress bar
            current_acc = 100 * correct_predictions / total_samples
            train_pbar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'Acc': f"{current_acc:.1f}%"
            })
        
        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc = 100 * correct_predictions / total_samples
        
        # ============ VALIDATION PHASE ============
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc="✅ Validation")
            for inputs, labels in val_pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                current_val_acc = 100 * val_correct / val_total
                val_pbar.set_postfix({
                    'Loss': f"{loss.item():.4f}",
                    'Acc': f"{current_val_acc:.1f}%"
                })
        
        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_acc = 100 * val_correct / val_total
        
        # Update scheduler
        scheduler.step(epoch_val_loss)
        
        # Save best model
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            best_model_state = model.state_dict().copy()
            print(f"🎉 NEW BEST MODEL! Validation Accuracy: {best_val_acc:.2f}%")
        
        # Store history
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        
        # Print summary
        print(f"\n📊 EPOCH {epoch+1} SUMMARY:")
        print(f"   Training   → Loss: {epoch_train_loss:.4f}, Acc: {epoch_train_acc:.2f}%")
        print(f"   Validation → Loss: {epoch_val_loss:.4f}, Acc: {epoch_val_acc:.2f}%")
        print(f"   Best Val Acc: {best_val_acc:.2f}%")
    
    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
        print(f"\n🏆 Training completed! Best accuracy: {best_val_acc:.2f}%")
    
    return model, history

# =====================================================
# EVALUATION FUNCTIONS
# =====================================================

def evaluate_model(model, data_loader):
    """Comprehensive model evaluation"""
    print("\n🔍 Evaluating model performance...")
    
    model.eval()
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probabilities = torch.softmax(outputs, dim=1)
            _, predictions = torch.max(outputs, 1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    # Calculate metrics
    cm = confusion_matrix(all_labels, all_predictions)
    report = classification_report(all_labels, all_predictions, 
                                 target_names=['Fake', 'Real'], digits=4)
    
    # Calculate detailed metrics
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, len(all_labels))
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print("\n" + "="*60)
    print("📊 EVALUATION RESULTS")
    print("="*60)
    print(f"🎯 Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"🔍 Precision: {precision:.4f}")
    print(f"📈 Recall: {recall:.4f}")
    print(f"⚖️ F1-Score: {f1_score:.4f}")
    print(f"\n📋 Classification Report:")
    print(report)
    print("="*60)
    
    return accuracy

def plot_training_history(history):
    """Plot training history"""
    plt.figure(figsize=(12, 5))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss', color='blue')
    plt.plot(history['val_loss'], label='Validation Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Training Accuracy', color='blue')
    plt.plot(history['val_acc'], label='Validation Accuracy', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

def save_model(model, optimizer=None, epoch=None, accuracy=None):
    """Save model with metadata"""
    
    save_dict = {
        'model_state_dict': model.state_dict(),
        'model_architecture': 'ResNet50',
        'config': {
            'img_size': Config.IMG_SIZE,
            'batch_size': Config.BATCH_SIZE,
            'learning_rate': Config.LEARNING_RATE,
            'dropout_rate': Config.DROPOUT_RATE,
            'num_epochs': Config.NUM_EPOCHS
        }
    }
    
    if optimizer:
        save_dict['optimizer_state_dict'] = optimizer.state_dict()
    if epoch:
        save_dict['epoch'] = epoch
    if accuracy:
        save_dict['accuracy'] = accuracy
    
    torch.save(save_dict, Config.MODEL_PATH)
    print(f"💾 Model saved to: {Config.MODEL_PATH}")

def load_model(model, model_path=None):
    """Load saved model"""
    if model_path is None:
        model_path = Config.MODEL_PATH
    
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Model loaded from: {model_path}")
        return True
    else:
        print(f"❌ Model file not found: {model_path}")
        return False

# =====================================================
# PREDICTION FUNCTION
# =====================================================

def predict_image(model, image_path):
    """Make prediction on a single image"""
    model.eval()
    
    try:
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = val_transform(image).unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            _, prediction = torch.max(outputs, 1)
        
        fake_prob = probabilities[0][0].item() * 100
        real_prob = probabilities[0][1].item() * 100
        pred_class = "Real" if prediction.item() == 1 else "Fake"
        confidence = max(fake_prob, real_prob)
        
        print(f"\n🔍 Prediction Results:")
        print(f"📸 Image: {image_path}")
        print(f"🎯 Prediction: {pred_class}")
        print(f"📊 Confidence: {confidence:.2f}%")
        print(f"📈 Real probability: {real_prob:.2f}%")
        print(f"📉 Fake probability: {fake_prob:.2f}%")
        
        return pred_class, confidence, fake_prob, real_prob
        
    except Exception as e:
        print(f"❌ Error making prediction: {e}")
        return None, None, None, None

# =====================================================
# MAIN EXECUTION
# =====================================================

def main():
    """Main execution function"""
    
    print("\n" + "="*80)
    print("🚀 PRODUCTION DEEPFAKE DETECTION TRAINING")
    print("🧠 ResNet50-based Deep Learning Model")
    print("="*80)
    
    try:
        # Validate paths
        if not os.path.exists(Config.REAL_PATH):
            print(f"❌ Real images path not found: {Config.REAL_PATH}")
            print("Please update Config.REAL_PATH with the correct path to your real images")
            return
        if not os.path.exists(Config.FAKE_PATH):
            print(f"❌ Fake images path not found: {Config.FAKE_PATH}")
            print("Please update Config.FAKE_PATH with the correct path to your fake images")
            return
        
        # Create data loaders
        train_loader, val_loader, class_weights = create_data_loaders()
        
        # Initialize model
        print("\n🏗️ Initializing model...")
        model = ProductionDeepfakeDetector(
            num_classes=2, dropout_rate=Config.DROPOUT_RATE
        ).to(device)
        
        # Setup training components
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.AdamW(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=1e-4
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-7
        )
        
        # Check for existing model
        if os.path.exists(Config.MODEL_PATH):
            response = input(f"Found existing model. Continue training? (y/n): ")
            if response.lower() == 'y':
                if load_model(model):
                    print("Continuing training from saved model...")
        
        # Train model
        print(f"\n🚀 Starting training...")
        model, history = train_model(
            model, train_loader, val_loader, criterion, optimizer, scheduler, Config.NUM_EPOCHS
        )
        
        # Plot training history
        plot_training_history(history)
        
        # Final evaluation
        accuracy = evaluate_model(model, val_loader)
        
        # Save model
        save_model(model, optimizer, accuracy=accuracy)
        
        print("\n🎉 TRAINING COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"🏆 Final Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"💾 Model saved to: {Config.MODEL_PATH}")
        print("="*80)
        
        # Test on a sample image (if exists)
        test_image = "test_image.jpg"  # Change this to your test image path
        if os.path.exists(test_image):
            print(f"\n🧪 Testing on sample image: {test_image}")
            predict_image(model, test_image)
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        if 'model' in locals():
            save_model(model, filename=Config.MODEL_PATH.replace('.pth', '_interrupted.pth'))
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()