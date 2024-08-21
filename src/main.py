# File: src/main.py

import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src', 'stable-fast-3d'))



import gradio as gr
from src.ui.gradio_interface import create_interface

def main():
    demo = create_interface()
    demo.launch()

if __name__ == "__main__":
    main()
    
    