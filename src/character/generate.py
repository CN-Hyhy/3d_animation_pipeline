# File: src/character/generate.py

import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
stable_fast_3d_path = os.path.join(project_root, 'src', 'stable_fast_3d')
sys.path.insert(0, project_root)
sys.path.insert(0, stable_fast_3d_path)

import torch
from sf3d.system import SF3D
from PIL import Image
import trimesh

class CharacterGenerator:
    def __init__(self):
        self.model = SF3D.from_pretrained(
            "stabilityai/stable-fast-3d",
            config_name="config.yaml",
            weight_name="model.safetensors"
        )
        self.model.to("cuda" if torch.cuda.is_available() else "cpu")

    def generate_character(self, prompt, num_inference_steps=50):
        # Tạo một ảnh từ prompt (đây là một placeholder, bạn cần thay thế bằng logic thực tế)
        image = Image.new('RGBA', (512, 512), (255, 255, 255, 255))
        
        # Sử dụng SF3D để tạo mesh
        mesh, global_dict = self.model.run_image(
            image,
            bake_resolution=1024,
            remesh="quad",
            estimate_illumination=True
        )
        
        return mesh

    def save_character(self, mesh, filename):
        if isinstance(mesh, trimesh.Trimesh):
            mesh.export(filename)
        else:
            raise ValueError("Mesh is not a trimesh.Trimesh object")

# Ví dụ sử dụng
if __name__ == "__main__":
    generator = CharacterGenerator()
    character_mesh = generator.generate_character("A stylized 3D character with big eyes and spiky hair")
    generator.save_character(character_mesh, "generated_character.obj")
    
        
        
