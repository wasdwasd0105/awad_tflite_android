import kivy
from kivy.app import App
from kivy.properties import NumericProperty
#from kivy.uix.boxlayout import BoxLayout
from kivy.garden.graph import Graph, LinePlot
from pandas.core.frame import DataFrame
import time
import datetime
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.button import MDRaisedButton
from kivy.clock import Clock, mainthread
from kivymd.app import MDApp
import os
import sys
from pathlib import Path
from itertools import chain
from threading import Thread
from kivymd.uix.screen import MDScreen
from src.dataloader import AWAD_Dataset_local
from tflite_runtime.interpreter import Interpreter
from multiprocessing import Process, Manager
from copy import deepcopy

class PlethPlot(MDScreen):

    zoom = NumericProperty(1)

    ymin = NumericProperty(29000)
    ymax = NumericProperty(38000)

    x_ori_min = NumericProperty(0)
    x_awad_min = NumericProperty(0)

    samples = 3000
    time = 0
    awad_x_enable = 0

    def go_back(self):
        self.time = 0
        self.awad_x_enable = 0
        self.shared_dict['total_val'] = 0
        self.shared_dict['total'] = 1
        self.shared_dict['isfinished'] = 0
        self.shared_dict['awad_res'] = []
        self.x_ori_min = 0
        self.x_awad_min = 0
        self.t.kill()
        self.ids.topbar.title = "Initing Please Wait"
        self.ids.modulation2.remove_widget(self.graph_pleth)
        self.ids.modulation.remove_widget(self.graph)
        self.parent.current = "start_screen"

  
    def start_plot(self):

        self.zoom = 1
        self.ymin = 29000
        self.ymax = 38000

        self.plot_graph1()
        self.plot_graph_pleth()

        # set timer, call every 1s
        self.timer = Clock.schedule_interval(self.timer_callback, 1)

        # use process manager to share contents between processes
        self.shared_manager = Manager()
        self.shared_dict = self.shared_manager.dict()
        self.shared_dict['total_val'] = 0
        self.shared_dict['total'] = 1
        self.shared_dict['isfinished'] = 0
        self.shared_dict['awad_res'] = []

        self.awad_res_list = self.shared_manager.list()

        self.ids.topbar.title = "Pleth Viewer: Running AWAD"

        # use a subprocess to run the tflite; it can use mutiple thread to use all cpu cores
        self.t = Process(target=self.pleth_awad, args=(123,))
        self.t.start()

    

    def plot_graph_pleth(self):
        self.graph_pleth = Graph(
                           xlabel='', ylabel='Pleth',
                           y_ticks_major=2000,
                           x_ticks_major=128,
                           label_options = {'color': [0, 0, 0, 0.7], 'bold': True},
                           border_color=[0, 1, 1, 1],
                           tick_color=[0, 0, 0, 0.7],
                           x_grid=True, y_grid=True, padding=5,
                           xmin=1, xmax=self.samples,
                           ymin=29000, ymax=38000,
                           draw_border=True,
                           x_grid_label=True, y_grid_label=False)
        
        self.graph_pleth.background_color = 1, 1, 1, 1
        self.ids.modulation2.add_widget(self.graph_pleth)
        
        self.plot_ori = LinePlot(color=[0, 0.25, 1, 1], line_width=1.5)
        self.graph_pleth.add_plot(self.plot_ori)

        # driectly plot the orginal pleth using dataloader
        inpsize = 512
        self.case_loader = AWAD_Dataset_local(os.path.join(os.environ["ROOT_DIR"],'Data_Different_Sets_PerPatient/'), 
            [os.environ["cur_case"]], len_seg = inpsize, group_amt = 9,isLabel = True)
        case_list = []
        for val_data in self.case_loader:
            pleth = val_data['pleth_ctr']
            case_list.append(pleth.tolist())
        self.pelth_ori = sum(case_list, [])
        self.pelth_ori = np.array(self.pelth_ori).astype(int)
        #print(self.pelth_ori)
        if self.samples > len(self.pelth_ori):
            self.ori_range = len(self.pelth_ori)
        else:
            self.ori_range = self.samples

        self.plot_ori.points = [(x, self.pelth_ori[x]) for x in range(self.ori_range)]

    def plot_graph1(self):
        self.graph = Graph(
                           xlabel='', ylabel='Clean Pleth',
                           y_ticks_major=2000,
                           x_ticks_major=128,
                           label_options = {'color': [0, 0, 0, 0.7], 'bold': True},
                           border_color=[0, 1, 1, 1],
                           tick_color=[0, 0, 0, 0.7],
                           x_grid=True, y_grid=True, padding=5,
                           xmin=0, xmax=self.samples,
                           ymin=29000, ymax=38000,
                           draw_border=True,
                           x_grid_label=True, y_grid_label=False)
        self.graph.background_color = 1, 1, 1, 1
        self.ids.modulation.add_widget(self.graph)
        #self.plot_ori = np.linspace(0, 1, self.samples)
        self.plot_awad = LinePlot(color=[0, 1, 0.25, 1], line_width=1.5)
        self.graph.add_plot(self.plot_awad)

        pass


    def timer_callback(self, dt):
        # Timer: call every 1s, and check if tflite finish
        self.time += 1
        self.ids.awad_status.text = "       " + "Case: " + os.environ["cur_case"] + "   Taking ML: "+ str(self.shared_dict['total_val']) + "/" + str(self.shared_dict['total']) + "    Time:" + str(self.time)

        if self.shared_dict['isfinished'] == 1:
            self.ids.topbar.title = "Outputting AWAD Pleth"
            self.timer.cancel()
            self.awad_pleth = deepcopy(self.shared_dict['awad_res'])
            if self.samples > len(self.awad_pleth):
                self.awad_range = len(self.awad_pleth)
                self.plot_awad.points = [(x, self.awad_pleth[x]) for x in range(len(self.awad_pleth))]
            else:
                self.awad_range = self.samples
                self.plot_awad.points = [(x, self.awad_pleth[x]) for x in range(self.awad_range)]
                self.awad_x_enable = 1
            
            #self.graph.xmax = len(awad_pleth)
            self.ids.topbar.title = "Pleth Viewer -- Finished"


    def pleth_awad(self,aa):
        
        res_hand = []
        res_foot = []
        inpsize = 512
        self.case_loader = AWAD_Dataset_local(os.path.join(os.environ["ROOT_DIR"],'Data_Different_Sets_PerPatient/'), 
            [os.environ["cur_case"]], len_seg = inpsize, group_amt = 9,isLabel = True)
        interpreter = Interpreter(model_path=os.environ["cur_model_path"] , num_threads=int(os.environ["Num_thread"]))
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        #print(output_details)
        correct = 0
        total = len(self.case_loader)
        total_val = 0
        case_list_pred = []
        self.shared_dict['total'] = total


        for val_data in self.case_loader:
            valseg, vallabels = val_data['segment'], val_data['gt']
            pleth = val_data['pleth_ctr']

            #reshape the array
            valseg = np.reshape(valseg, (1, 1, inpsize))

            # set tenser and predict
            interpreter.set_tensor(input_details[0]['index'], valseg)
            interpreter.invoke()
            valoutputs = interpreter.get_tensor(output_details[0]['index'])
            #print(valoutputs)
            valpredicted = np.argmax(valoutputs.data, 1)
            if valpredicted == [0]:
                case_list_pred.append(pleth.tolist())
                # TODO pleth has 9 segs but we only need the center one, 
                # save and return pleth_ctr to the plotter
            total_val += 1
            self.shared_dict['total_val'] = total_val

        self.shared_dict['awad_res'] = sum(case_list_pred, [])
        #print(self.shared_dict['awad_res'])
        self.shared_dict['isfinished'] = 1


    def update_ymin(self, value):
        if value == '+' and self.ymin < 50000:
            self.ymin += 1000
        elif value == '-' and self.ymin > 20000:
            self.ymin -= 1000
        self.graph.ymin = self.ymin
        self.graph_pleth.ymin = self.ymin

    def update_ymax(self, value):
        if value == '+' and self.ymax < 50000:
            self.ymax += 1001
        elif value == '-' and self.ymax > 20000:
            self.ymax -= 1001
        self.graph.ymax = self.ymax
        self.graph_pleth.ymax = self.ymax

    def update_x_ori(self, value):
        if value == '+' and self.x_ori_min + self.ori_range < len(self.pelth_ori):
            if len(self.pelth_ori) - self.x_ori_min < self.ori_range*2:
                self.x_ori_min += self.samples
                self.plot_ori.points = [(x - self.x_ori_min, self.pelth_ori[x]) for x in range(self.x_ori_min, len(self.pelth_ori))]
            else:
                self.x_ori_min += self.samples
                self.plot_ori.points = [(x - self.x_ori_min, self.pelth_ori[x]) for x in range(self.x_ori_min, self.x_ori_min + self.ori_range)]

        elif value == '-' and self.x_ori_min > 0:
            self.x_ori_min -= self.samples
            self.plot_ori.points = [(x - self.x_ori_min, self.pelth_ori[x]) for x in range(self.x_ori_min, self.x_ori_min + self.ori_range)]

    def update_x_awad(self, value):
        if self.awad_x_enable == 1 and value == '+' and self.x_awad_min + self.awad_range < len(self.awad_pleth):
            if len(self.awad_pleth) - self.x_awad_min < self.awad_range*2:
                self.x_awad_min += self.samples
                self.plot_awad.points = [(x - self.x_awad_min, self.awad_pleth[x]) for x in range(self.x_awad_min, len(self.awad_pleth))]
            else:
                self.x_awad_min += self.samples
                self.plot_awad.points = [(x - self.x_awad_min, self.awad_pleth[x]) for x in range(self.x_awad_min, self.x_awad_min + self.awad_range)]

        elif self.awad_x_enable == 1 and value == '-' and self.x_awad_min > 0:
            self.x_awad_min -= self.samples
            self.plot_awad.points = [(x - self.x_awad_min, self.awad_pleth[x]) for x in range(self.x_awad_min, self.x_awad_min + self.awad_range)]


    def update_zoom(self, value):
        if value == '+' and self.zoom < 16:
            self.zoom *= 2
            self.graph.x_ticks_major /= 2
        elif value == '-' and self.zoom > 1:
            self.zoom /= 2
            self.graph.x_ticks_major *= 2