import os
import sys
from pathlib import Path

from kivy.uix.screenmanager import Screen
from kivy.lang import Builder
from kivymd.app import MDApp
from kivymd.uix.button import MDRectangleFlatButton
from kivy.core.window import Window
from kivy.utils import platform

os.environ["VERSION"] = " Ver 1.3"

if getattr(sys, "frozen", False):  # bundle mode with PyInstaller
    os.environ["ROOT_DIR"] = sys._MEIPASS
else:
    sys.path.append(os.path.abspath(__file__).split("demos")[0])
    os.environ["ROOT_DIR"] = str(Path(__file__).parent)


os.environ["Model_dir"] = os.path.join(os.environ["ROOT_DIR"],'AWAD_tflite')


Window.softinput_mode = "below_target"
class MainApp(MDApp):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.theme_cls.material_style = "M2"
        self.theme_cls.primary_palette = "Cyan"
        self.theme_cls.theme_style == "Dark"

    
    def build(self):
        Builder.load_file(
            os.path.join(
                os.environ["ROOT_DIR"], "src", "start_screen.kv"
            )
        )
        Builder.load_file(
            os.path.join(
                os.environ["ROOT_DIR"], "src", "pleth_viewer.kv"
            )
        )
        Builder.load_file(
            os.path.join(
                os.environ["ROOT_DIR"], "src", "full_val.kv"
            )
        )

        return Builder.load_file(
            os.path.join(
                os.environ["ROOT_DIR"], "src", "screen_manager.kv"
            )
        )


MainApp().run()