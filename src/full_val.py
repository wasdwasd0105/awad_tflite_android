import os
from threading import Thread
from kivymd.uix.screen import MDScreen
from kivy.utils import platform
from src.dataloader import AWAD_Dataset_local
from tflite_runtime.interpreter import Interpreter
import numpy as np
from kivy.clock import Clock, mainthread
import time 
from multiprocessing import Process, Manager

class FullVal(MDScreen):
    result_output = {}
    extracted_features_path = os.environ["ROOT_DIR"]
    val_cases = ['N1001', 'N1022','N1032', 'X1001', 'X3005', 'X4001', 'N1048', 'N3002']
    #val_cases = ['N1001']

    time = 0

    def start_test(self):
        self.ids.model_bar.title = "Val Benchmark: " + str(os.environ["cur_model"])
        self.ids.cases_bar.title = "Cases: " + str(self.val_cases)
        # if platform == 'android':
        #     from jnius import autoclass
        #     WindowManager = autoclass('android.view.WindowManager$LayoutParams')
        #     activity = autoclass('org.kivy.android.PythonActivity').mActivity
        #     window = activity.getWindow()
        #     window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

        self.timer = Clock.schedule_interval(self.timer_callback, 1)
        self.ids.awad_status.text = "Running Dataloder..."
        self.ids.progress_determinate.value = 10
        self.shared_manager = Manager()
        self.shared_dict = self.shared_manager.dict()
        self.shared_dict['total_val'] = 0
        self.shared_dict['total'] = 1
        self.shared_dict['isfinished'] = 0
        self.shared_dict['acc'] = 0

        self.t = Process(target=self.run_thread, args=(123,))
        self.t.start()

      
    def go_back(self):
        self.timer = 0
        self.t.kill()
        self.parent.current = "start_screen"
        
        
    def timer_callback(self, dt):
        self.time += 1
        self.ids.timer_bar.title = "Timer: " + str(self.time) + 's'
        self.ids.awad_status.text = "Taking ML: "+ str(self.shared_dict['total_val']) + "/" + str(self.shared_dict['total'])
        self.ids.progress_determinate.value = 10 + int(100*(self.shared_dict['total_val']/self.shared_dict['total']))
        if self.shared_dict['isfinished'] == 1:
            self.ids.timer_bar.title = "Finished. Time: " + str(self.shared_dict['time'])
            self.ids.awad_status.text = "Taking ML: "+ str(self.shared_dict['total_val']) + "/" + str(self.shared_dict['total']) + "    ACC" + str(self.shared_dict['acc'])
            self.timer.cancel()
            self.ids.progress_determinate.color = 0, 0.9, 0.1, 1
            self.ids.progress_determinate.value = 100
            self.ids.spinner.active = False


    def run_thread(self, a):
        
        inpsize = 512
        bsize = 100
        
        start = time.time()

        val_dataset = AWAD_Dataset_local(os.path.join(os.environ["ROOT_DIR"],'Data_Different_Sets_PerPatient/'), self.val_cases, len_seg = inpsize, group_amt = 9,isLabel = True)

        
        # load tflite model
        interpreter = Interpreter(model_path=os.environ["cur_model_path"] , num_threads=int(os.environ["Num_thread"]))
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(output_details)
        correct = 0
        total = len(val_dataset)
        predicted_full = []
        labels_full = []
        total_val = 0
        total_correct = 0
        val_predicted_full = []
        val_labels_full = []


        for val_data in val_dataset:
            valseg, vallabels = val_data['segment'], val_data['gt']
            
            print("Taking ML: "+ str(total_val) + "/" + str(total))
            self.shared_dict['total_val'] = total_val
            self.shared_dict['total'] = total

            #reshape the array
            valseg = np.reshape(valseg, (1, 1, inpsize))

            # set tenser and predict
            interpreter.set_tensor(input_details[0]['index'], valseg)
            interpreter.invoke()
            valoutputs = interpreter.get_tensor(output_details[0]['index'])

            #print(valoutputs)
            valpredicted = np.argmax(valoutputs.data, 1)
            #print(valpredicted)

            total_val += 1
            total_correct += (valpredicted == vallabels).sum().item()
            val_predicted_full.append(valpredicted)
            val_labels_full.append(vallabels)


        
        #print("  --  Validation Acc = ", (total_correct/total_val))
        end = time.time()
        self.shared_dict['isfinished'] = 1
        self.shared_dict['acc'] = total_correct/total_val
        self.shared_dict['time'] = end - start
        print(end - start)

