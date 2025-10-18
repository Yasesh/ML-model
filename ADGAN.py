# ADGAN_v2_Stable.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, utils
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm
import tarfile
from PIL import Image  # Ensure PIL is imported for anomaly detection part

# ------------------------
# Settings
# ------------------------
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CATEGORY = 'pill'
DATASET_ARCHIVE = "/Users/yazzyyy/Downloads/mvtec_anomaly_detection.tar.xz"
DATASET_PATH = "/Users/yazzyyy/Downloads/mvtec_anomaly_detection"

IMG_SIZE = 64
BATCH_SIZE = 16
EPOCHS = 100
LATENT_DIM = 100

# ------------------------
# Extract dataset if needed
# ------------------------
if not os.path.exists(DATASET_PATH):
    print("Extracting dataset...")
    with tarfile.open(DATASET_ARCHIVE) as tar:
        # Use shutil.unpack_archive or similar if tarfile has issues,
        # but filter=None is correct for modern Python
        tar.extractall(path=os.path.dirname(DATASET_ARCHIVE), filter=None)
    print("Dataset extracted.")
else:
    print("Dataset already extracted.")

TRAIN_PATH = os.path.join(DATASET_PATH, CATEGORY, "train", "good")
TEST_PATH = os.path.join(DATASET_PATH, CATEGORY, "test")

# ------------------------
# Data loader (with normalization)
# ------------------------
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # normalize to [-1, 1]
])

# The root directory must be the parent of the class folder ('good')
train_dataset = datasets.ImageFolder(root=os.path.dirname(os.path.dirname(TRAIN_PATH)), transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)


# ------------------------
# DCGAN Generator
# ------------------------
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            # Input: LATENT_DIM x 1 x 1 -> 4x4
            nn.ConvTranspose2d(LATENT_DIM, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            # 4x4 -> 8x8
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            # 8x8 -> 16x16
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            # 16x16 -> 32x32
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            # 32x32 -> 64x64
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)


# ------------------------
# DCGAN Discriminator
# ------------------------
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            # 64x64 -> 32x32
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # 32x32 -> 16x16
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # 16x16 -> 8x8
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # 8x8 -> 4x4
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            # 4x4 -> 1x1 (Final convolution to get one output per image)
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            # *** FIX: REMOVED nn.Sigmoid() for BCEWithLogitsLoss stability ***
        )

    def forward(self, x):
        # Output is (N, 1, 1, 1), .view(-1, 1) converts it to (N, 1) of logits
        return self.main(x).view(-1, 1)


G = Generator().to(DEVICE)
D = Discriminator().to(DEVICE)
criterion = nn.BCEWithLogitsLoss()
optimizerD = optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerG = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))


# ------------------------
# Train DCGAN
# ------------------------
def train():
    for epoch in range(EPOCHS):
        for i, (imgs, _) in enumerate(tqdm(train_loader, total=len(train_loader))):
            imgs = imgs.to(DEVICE)
            bs = imgs.size(0)

            # Labels remain the same (0/1) for BCEWithLogitsLoss
            real_labels = torch.ones(bs, 1).to(DEVICE)
            fake_labels = torch.zeros(bs, 1).to(DEVICE)

            # Train D
            D.zero_grad()

            # 1. Real Batch
            outputs = D(imgs)
            d_loss_real = criterion(outputs, real_labels)

            # 2. Fake Batch
            z = torch.randn(bs, LATENT_DIM, 1, 1).to(DEVICE)
            fake_imgs = G(z)
            outputs = D(fake_imgs.detach())
            d_loss_fake = criterion(outputs, fake_labels)

            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optimizerD.step()

            # Train G
            G.zero_grad()
            outputs = D(fake_imgs)
            # Generator wants discriminator to output 1 (real) for fake images
            g_loss = criterion(outputs, real_labels)
            g_loss.backward()
            optimizerG.step()

        print(f"Epoch [{epoch + 1}/{EPOCHS}] D_loss: {d_loss.item():.4f}, G_loss: {g_loss.item():.4f}")

        # Save generated samples for visual check
        sample_z = torch.randn(16, LATENT_DIM, 1, 1).to(DEVICE)
        gen_imgs = G(sample_z)
        utils.save_image(gen_imgs, f"generated_epoch_{epoch + 1}.png", normalize=True, nrow=4)


# ------------------------
# Anomaly Detection
# ------------------------
def detect_anomalies(num_images=5):
    output_folder = os.path.join(os.getcwd(), "anomaly_results_v2")
    os.makedirs(output_folder, exist_ok=True)

    test_imgs = []
    # Find all test images recursively
    for root, _, files in os.walk(TEST_PATH):
        for f in files:
            if f.endswith(('.png', '.jpg', '.jpeg')):
                test_imgs.append(os.path.join(root, f))

    if not test_imgs:
        print("Warning: No test images found. Skipping anomaly detection.")
        return

    for img_file in test_imgs[:num_images]:
        img = Image.open(img_file).convert('RGB')

        # Apply transformation
        img_tensor = transform(img).unsqueeze(0).to(DEVICE)

        # Simple one-shot reconstruction (NOT proper AnoGAN optimization)
        G.eval()
        with torch.no_grad():
            z = torch.randn(1, LATENT_DIM, 1, 1).to(DEVICE)
            fake_img = G(z)
        G.train()

        # Calculate Residual/Anomaly Map (L1 difference)
        anomaly_map = torch.mean(torch.abs(img_tensor - fake_img), dim=1).squeeze().cpu().detach().numpy()

        # Un-normalize for visualization
        img_vis = ((img_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0) * 0.5) + 0.5).clip(0, 1)

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 3, 1)
        plt.title("Original")
        plt.imshow(img_vis)
        plt.axis('off')

        # Display the simple reconstruction
        fake_img_vis = ((fake_img.squeeze(0).cpu().numpy().transpose(1, 2, 0) * 0.5) + 0.5).clip(0, 1)
        plt.subplot(1, 3, 2)
        plt.title("Simple G Reconstruction")
        plt.imshow(fake_img_vis)
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.title("Anomaly Map (L1)")
        plt.imshow(anomaly_map, cmap='jet')
        plt.axis('off')

        plt.savefig(os.path.join(output_folder, f"anomaly_{os.path.basename(img_file)}"), bbox_inches='tight')
        plt.close()
        print(f"Saved anomaly map for: {img_file}")


# ------------------------
# Run
# ------------------------
if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    print("Starting DCGAN training with BCEWithLogitsLoss...")
    train()
    print("Training done. Detecting anomalies...")
    detect_anomalies()
    print("All done!")