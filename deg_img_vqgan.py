import os
import argparse
import torch
from torchvision import utils as vutils
from tqdm import tqdm
from vqgan import VQGAN
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class ImageDataset(Dataset):
    def __init__(self, file_path, transform=None):
        with open(file_path, 'r') as file:
            self.image_paths = file.readlines()
        self.image_paths = [path.strip() for path in self.image_paths]
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, img_path

# Define a transform to apply to the images (e.g., resize and convert to tensor)
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

def load_model(args):
    model = VQGAN(args).to(device=args.device)
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(checkpoint)
    model.eval()
    return model

def save_generated_image(input_image_path, generated_image, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    image_name = os.path.basename(input_image_path)
    generated_image_name = f"{os.path.splitext(image_name)[0]}_generated{os.path.splitext(image_name)[1]}"
    generated_image_path = os.path.join(output_dir, generated_image_name)
    vutils.save_image(generated_image.add(1).mul(0.5), generated_image_path)
    return generated_image_path

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # Load model
    vqgan = load_model(args)
    
    # Load data
    file_path = args.file_path  # Path to your file containing image paths
    dataset = ImageDataset(file_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Create output directories and files for mapping
    output_dir = "inference_results"
    original_paths_file = "original_val_paths.txt"
    generated_paths_file = "gen_val_paths.txt"
    
    os.makedirs(output_dir, exist_ok=True)
    
    with open(original_paths_file, 'w') as orig_file, open(generated_paths_file, 'w') as gen_file:
        # Inference
        with torch.no_grad():
            for batch_idx, (imgs, img_paths) in tqdm(enumerate(dataloader), total=len(dataloader)):
                imgs = imgs.to(device)
                decoded_images, _, _ = vqgan(imgs)
                for i, img_path in enumerate(img_paths):
                    generated_image_path = save_generated_image(img_path, decoded_images[i], output_dir)
                    orig_file.write(f"{img_path}\n")
                    gen_file.write(f"{generated_image_path}\n")
    
    print(f"Inference completed. Generated images and paths are saved in {output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VQGAN Inference")
    parser.add_argument('--latent-dim', type=int, default=256, help='Latent dimension n_z (default: 256)')
    parser.add_argument('--image-size', type=int, default=256, help='Image height and width (default: 256)')
    parser.add_argument('--num-codebook-vectors', type=int, default=1024, help='Number of codebook vectors (default: 256)')
    parser.add_argument('--beta', type=float, default=0.25, help='Commitment loss scalar (default: 0.25)')
    parser.add_argument('--image-channels', type=int, default=3, help='Number of channels of images (default: 3)')
    parser.add_argument('--file-path', type=str, default='files_shuf_train.list', help='Path to data (default: /data)')
    parser.add_argument('--device', type=str, default="cuda", help='Which device the training is on')
    parser.add_argument('--batch-size', type=int, default=16, help='Input batch size for training (default: 6)')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the model checkpoint')
    
    args = parser.parse_args()
    main(args)