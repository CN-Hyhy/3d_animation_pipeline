import gradio as gr
import sys
import os
import torch
import numpy as np
from PIL import Image
from functools import lru_cache
import tempfile
from gradio_litmodel3d import LitModel3D
import rembg

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src', 'stable_fast_3d'))

from sf3d.system import SF3D
import sf3d.utils as sf3d_utils
from src.character.generate import CharacterGenerator

# Cấu hình và khởi tạo model
COND_WIDTH, COND_HEIGHT = 512, 512
COND_DISTANCE = 1.6
COND_FOVY_DEG = 40
BACKGROUND_COLOR = [0.5, 0.5, 0.5]

c2w_cond = sf3d_utils.default_cond_c2w(COND_DISTANCE)
intrinsic, intrinsic_normed_cond = sf3d_utils.create_intrinsic_from_fov_deg(
    COND_FOVY_DEG, COND_HEIGHT, COND_WIDTH
)

rembg_session = rembg.new_session()

model = SF3D.from_pretrained(
    "stabilityai/stable-fast-3d",
    config_name="config.yaml",
    weight_name="model.safetensors",
)
model.eval()
model = model.cuda()

character_generator = CharacterGenerator()

# Các hàm tiện ích và xử lý (giữ nguyên từ phiên bản trước)
# Các hàm tiện ích và xử lý
@lru_cache
def checkerboard(squares: int, size: int, min_value: float = 0.5):
    base = np.zeros((squares, squares)) + min_value
    base[1::2, ::2] = 1
    base[::2, 1::2] = 1
    repeat_mult = size // squares
    return (base.repeat(repeat_mult, axis=0).repeat(repeat_mult, axis=1)[:size, :size, None].repeat(3, axis=-1))

def show_mask_img(input_image: Image) -> Image:
    img_numpy = np.array(input_image)
    h, w = img_numpy.shape[:2]
    alpha = img_numpy[:, :, 3] / 255.0
    chkb = checkerboard(32, max(h, w))[:h, :w]
    new_img = img_numpy[..., :3] * alpha[:, :, None] + chkb * (1 - alpha[:, :, None])
    return Image.fromarray(new_img.astype(np.uint8), mode="RGB")

def remove_background(input_image: Image) -> Image:
    return rembg.remove(input_image, session=rembg_session)

def create_batch(input_image: Image) -> dict:
    img_cond = torch.from_numpy(np.asarray(input_image.resize((COND_WIDTH, COND_HEIGHT))).astype(np.float32) / 255.0).float().clip(0, 1)
    mask_cond = img_cond[:, :, -1:]
    rgb_cond = torch.lerp(torch.tensor(BACKGROUND_COLOR)[None, None, :], img_cond[:, :, :3], mask_cond)
    batch_elem = {
        "rgb_cond": rgb_cond,
        "mask_cond": mask_cond,
        "c2w_cond": c2w_cond.unsqueeze(0),
        "intrinsic_cond": intrinsic.unsqueeze(0),
        "intrinsic_normed_cond": intrinsic_normed_cond.unsqueeze(0),
    }
    return {k: v.unsqueeze(0) for k, v in batch_elem.items()}

def run_model(input_image, remesh_option, texture_size):
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            model_batch = create_batch(input_image)
            model_batch = {k: v.cuda() for k, v in model_batch.items()}
            trimesh_mesh, _glob_dict = model.generate_mesh(model_batch, texture_size, remesh_option.lower())
            trimesh_mesh = trimesh_mesh[0]
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".glb")
    trimesh_mesh.export(tmp_file.name, file_type="glb", include_normals=True)
    return tmp_file.name

# Hàm xử lý cho nút Run
def run_button(input_image, remesh_option, texture_size):
    glb_file = run_model(input_image, remesh_option.lower(), texture_size)
    return gr.update(value=glb_file, visible=True), gr.update(visible=True)

# Các hàm từ giao diện cũ
def generate_3d_character(character_description):
    try:
        mesh = character_generator.generate_character(character_description)
        output_file = f"generated_character_{hash(character_description)}.obj"
        character_generator.save_character(mesh, output_file)
        return f"3D character generated and saved as {output_file}"
    except Exception as e:
        return f"Error generating character: {str(e)}"

def apply_rigging(character_model):
    return f"Rigging applied to: {character_model}"

def generate_scene(scene_description):
    return f"Scene generated based on: {scene_description}"

def render_scene(scene):
    return f"Scene rendered: {scene}"

def sync_lip_with_audio(character_model, audio_file):
    return f"Lip synced for {character_model} with {audio_file}"

def composite_layers(rendered_scene, additional_layers):
    return f"Composited {rendered_scene} with {additional_layers}"

def denoise_rendered_scene(rendered_scene):
    return f"Denoised: {rendered_scene}"

# Custom CSS (giữ nguyên từ phiên bản cũ)
custom_css = """
body {
    background-color: #2b2b2b;
    color: #e0e0e0;
}
h1 {
    color: #ff9900 !important;
}
.gr-button {
    background-color: #4CAF50 !important;
    border-color: #4CAF50 !important;
}
.gr-button:hover {
    background-color: #45a049 !important;
    border-color: #45a049 !important;
}
.gr-form {
    background-color: #333333;
    border-radius: 10px;
    padding: 20px;
    margin-bottom: 20px;
}
.gr-input, .gr-textarea {
    background-color: #444444 !important;
    color: #e0e0e0 !important;
    border-color: #555555 !important;
}
.gr-input:focus, .gr-textarea:focus {
    border-color: #ff9900 !important;
}
.gr-text-input {
    color: #e0e0e0 !important;
}
"""

def create_interface():
    with gr.Blocks(css=custom_css) as demo:
        gr.Markdown("# 3D Animation Pipeline")
        
        with gr.Tab("Character Creation"):
            with gr.Group():
                input_img = gr.Image(type="pil", label="Input Image", sources="upload", image_mode="RGBA")
                preview_removal = gr.Image(label="Preview", type="pil", image_mode="RGB", interactive=False)
                
                remesh_option = gr.Radio(choices=["None", "Triangle", "Quad"], label="Remeshing", value="None")
                texture_size = gr.Slider(label="Texture Size", minimum=512, maximum=2048, value=1024, step=256)
                
                run_btn = gr.Button("Generate 3D Character (Image-based)")
                
                output_3d = LitModel3D(label="3D Model", visible=False, clear_color=[0.0, 0.0, 0.0, 0.0], tonemapping="aces", contrast=1.0, scale=1.0)
            
            with gr.Group():
                char_input = gr.Textbox(label="Character Description")
                char_output = gr.Textbox(label="Generated Character (Text-based)")
                char_button = gr.Button("Generate 3D Character (Text-based)")
            
            with gr.Group():
                rigging_input = gr.Textbox(label="Character Model")
                rigging_output = gr.Textbox(label="Rigged Character")
                rigging_button = gr.Button("Apply Rigging")

        with gr.Tab("Scene Generation"):
            with gr.Group():
                scene_input = gr.Textbox(label="Scene Description")
                scene_output = gr.Textbox(label="Generated Scene")
                scene_button = gr.Button("Generate Scene")
            
            with gr.Group():
                render_input = gr.Textbox(label="Scene to Render")
                render_output = gr.Textbox(label="Rendered Scene")
                render_button = gr.Button("Render Scene")

        with gr.Tab("Post-processing"):
            with gr.Group():
                lip_sync_char = gr.Textbox(label="Character Model")
                lip_sync_audio = gr.Audio(label="Audio File")
                lip_sync_output = gr.Textbox(label="Lip Sync Result")
                lip_sync_button = gr.Button("Sync Lip with Audio")
            
            with gr.Group():
                composite_scene_input = gr.Textbox(label="Rendered Scene")
                composite_layers_input = gr.Textbox(label="Additional Layers")
                composite_output = gr.Textbox(label="Composited Result")
                composite_button = gr.Button("Composite Layers")
            
            with gr.Group():
                denoise_input = gr.Textbox(label="Rendered Scene to Denoise")
                denoise_output = gr.Textbox(label="Denoised Scene")
                denoise_button = gr.Button("Denoise Rendered Scene")

        # Event handlers
        input_img.change(
            lambda x: gr.update(value=show_mask_img(remove_background(x)), visible=True) if x is not None else gr.update(visible=False),
            inputs=[input_img],
            outputs=[preview_removal]
        )
        
        run_btn.click(run_button, inputs=[input_img, remesh_option, texture_size], outputs=[output_3d, output_3d])
        
        char_button.click(generate_3d_character, inputs=char_input, outputs=char_output)
        rigging_button.click(apply_rigging, inputs=rigging_input, outputs=rigging_output)
        scene_button.click(generate_scene, inputs=scene_input, outputs=scene_output)
        render_button.click(render_scene, inputs=render_input, outputs=render_output)
        lip_sync_button.click(sync_lip_with_audio, inputs=[lip_sync_char, lip_sync_audio], outputs=lip_sync_output)
        composite_button.click(composite_layers, inputs=[composite_scene_input, composite_layers_input], outputs=composite_output)
        denoise_button.click(denoise_rendered_scene, inputs=denoise_input, outputs=denoise_output)

    return demo