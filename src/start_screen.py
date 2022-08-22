from kivymd.uix.screen import MDScreen
import os
from kivy.utils import platform
from kivymd.uix.datatables import MDDataTable
from kivymd.uix.menu import MDDropdownMenu
from kivy.metrics import dp
from kivy.uix.widget import Widget
from kivymd.uix.label import MDLabel


from kivymd.uix.button import MDRaisedButton


class StartScreen(MDScreen):
    data_location = "internal"
    os.environ["Num_thread"] = '8'

    def checkbox(self, app, text, value):
        if platform == 'android':
            if text == "Internal data folder" and self.data_location != "internal":
                self.data_location = "internal"
                os.environ["Model_dir"] = os.path.join(os.environ["ROOT_DIR"],'AWAD_tflite')
                
            if text == "SDcard data folder(/sdcard/AWAD_tflite)" and self.data_location != "external":
                from android.permissions import request_permissions, Permission
                request_permissions([Permission.WRITE_EXTERNAL_STORAGE, Permission.READ_EXTERNAL_STORAGE])
                from android.storage import primary_external_storage_path
                self.data_location = "external"
                os.environ["Model_dir"] = os.path.join(primary_external_storage_path(), "AWAD_tflite")
                


    def choose_case(self, table, cur_row):
        os.environ["cur_case"] = cur_row[1]
        self.ids.caseid.text = os.environ["cur_case"]

    def choose_models(self, table, cur_row):
        os.environ["cur_model"] = cur_row[1]
        os.environ["cur_model_path"] = os.path.join(os.environ["Model_dir"], os.environ["cur_model"])
        self.ids.takeid.text = os.environ["cur_model"]

    def read_models_folder(self):
        model_files = os.listdir(os.environ["Model_dir"])
        cases_list = []
        for i in model_files:
            case = i
            if case[0] != '.' and case not in cases_list:
                cases_list.append(case)

        #print(len(cases_list))
        self.data_tables = MDDataTable(
        rows_num=100,
        check = True,
        size_hint=(0.8, 0.7),
        pos_hint={"center_x": 0.5, "center_y": 0.5},
        column_data=[
                ("No.", dp(20)),
                ("Model", dp(50)),
            ],
        row_data=[
            (f"{i + 1}", cases_list[i]) for i in range(len(cases_list))
            ],
        )
        self.data_tables.bind(on_check_press=self.choose_models)

        self.data_tables.ids.container.add_widget(
            Widget(size_hint_y=None, height="5dp")
        )
        self.data_tables.ids.container.add_widget(       
            MDRaisedButton(
                text="Confirm",
                pos_hint={"right": 1},
                on_release=lambda x: self.remove_widget(self.data_tables),
            )
        )
        self.data_tables.ids.container.add_widget(
            MDLabel(
                text="please only choose one model",
                size_hint=(0.5, 0.05),
                pos_hint={"left": 1},
            ),
        )
        self.add_widget(self.data_tables)
        
    def read_case_folder(self):

        cases_list = ['N1001', 'N1022','N1032', 'X1001', 'X3005', 'X4001', 'N1048', 'N3002']
        self.data_tables = MDDataTable(
        rows_num=100,
        check = True,
        size_hint=(0.8, 0.7),
        pos_hint={"center_x": 0.5, "center_y": 0.5},
        column_data=[
                ("No.", dp(20)),
                ("Case", dp(50)),
            ],
        row_data=[
            (f"{i + 1}", cases_list[i]) for i in range(len(cases_list))
            ],
        )
        self.data_tables.bind(on_check_press=self.choose_case)

        self.data_tables.ids.container.add_widget(
            Widget(size_hint_y=None, height="5dp")
        )
        self.data_tables.ids.container.add_widget(       
            MDRaisedButton(
                text="Confirm",
                pos_hint={"right": 1},
                on_release=lambda x: self.remove_widget(self.data_tables),
            )
        )
        self.data_tables.ids.container.add_widget(
            MDLabel(
                text="please only choose one case",
                size_hint=(0.5, 0.05),
                pos_hint={"left": 1},
            ),
        )
        self.add_widget(self.data_tables)


    def choose_num_thread(self):
        thread_items = [
            {
                "viewclass": "OneLineListItem",
                "text": f"{i} thread(s)",
                "height": dp(56),
                "on_release": lambda x=i: self.set_thread_munber(x),
            } for i in [1,2,4,8]
        ]
        self.menu = MDDropdownMenu(
            items=thread_items,
            caller=self.ids.threadid,
            width_mult=4,
        )
        self.menu.open()

    def set_thread_munber(self, x):
        os.environ["Num_thread"] = str(x)
        self.ids.threadid.text = str(x) + ' thread(s)'
        self.menu.dismiss()
