
import pickle
import pandas as pd
import numpy as np
import os
import random
from scipy.signal import find_peaks


class AWAD_Dataset_local():
    """Artifacts dataset."""

    def __init__(self, case_folder, cases_to_parse, len_seg=256, group_amt=5, case_num = None, isForecasting = False, isFixmatch = False, isLabel = False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            case_folder (string): Path to the folder with all pleth data
            len_pleth (int): Length pleths will be padded to
            group_amt(int): Amount of pleths grouped together
            transform (callable, optional): Initializing anything other than None will transform pleth into tensor

        """

        annotator1 = os.path.join(os.environ["ROOT_DIR"], "correct_label", "AnnotatorDict_Diana.pkl")
        annotator2 = os.path.join(os.environ["ROOT_DIR"], "correct_label", "AnnotatorDict_Anushri.pkl")

        with open(annotator1, 'rb') as f:
            self.annot_dict1 = pickle.load(f)
            
        with open(annotator2, 'rb') as f:
            self.annot_dict2 = pickle.load(f)
            
        self.case_folder = case_folder
        self.cases_to_parse = cases_to_parse
        self.len_seg = len_seg
        self.group_amt = group_amt
        self.isLabel = isLabel
        self.isForecasting = isForecasting
        self.isFixmatch = isFixmatch
        
        self.case_num = case_num
        
        if case_num != None:
            self.amt_0 = int(case_num/2)
            self.amt_1 = int(case_num/2)
        
        self.end = False
        self.ct = -1
        
        self.pleths = self.get_cases(case_folder,cases_to_parse)
        self.beats = self.get_beat(self.pleths)
        
        if self.isLabel:
            self.gtdict = self.gt_dict()
            
        if self.group_amt%2 == 0:
            self.group_amt = self.group_amt + 1
            

     
    ### Gotta fix here
    def __len__(self):
        if self.isLabel:
            ret = len(self.gtdict)
        else:
            ret = len(self.beats)
            
        #if self.case_num != None:
        #    ret = self.case_num
        
        return ret
    
    def artifact_getter(self,time,annotation):
        artifact_axis = []
        for i in range(len(time)):
            ##### !!!! CHANGED TO REFLECT OLD METHOD BEING APPLIED
            append_num = 1
            for arti_interval in annotation:
                if time[i] >= arti_interval[0] and time[i] <= arti_interval[1]:
                    append_num = 0
            artifact_axis.append(append_num)
        return artifact_axis
    
    def sum_artifact_axis(self,anshuri,diana, amount_of_1, amount_of_2):
        artifact_axis = []
        if len(diana) != len(anshuri):
            print("Something wrong with artifact axis")
        for i in range(len(diana)):
            num = diana[i]+anshuri[i]
            if num == 1:
                amount_of_1 = amount_of_1 + 1
            elif num == 2:
                amount_of_2 = amount_of_2 + 1
            artifact_axis.append(num)
        return artifact_axis, amount_of_1, amount_of_2
    
    def gt_dict(self):
        disa_count = 0
        agr_count = 0
        gtdict = {}
        for key in self.beats:
            
            if not(self.case_num == None):
                if self.amt_0 == 0 and self.amt_1 == 0:
                    self.end = True
                    break
            
            if self.end == True:
                break

            slice_df = self.beats[key].reset_index(drop=True)
            x_axis = slice_df['time'].tolist()
            
            annot_key = key.split('_')[0]+'_'+key.split('_')[1]+'_'+key.split('_')[2]
            #artifact_axis_dia = self.artifact_getter(x_axis,artifact_times_DIA[art_key])
            #artifact_axis_ans = self.artifact_getter(x_axis,artifact_times_ANS[art_key])
            try:
                artifact_axis_dia = self.artifact_getter(x_axis,self.annot_dict1[annot_key])
                artifact_axis_ans = self.artifact_getter(x_axis,self.annot_dict2[annot_key])

                artifact_axis, disa_count, agr_count = self.sum_artifact_axis(artifact_axis_ans, artifact_axis_dia, disa_count, agr_count)

                if max(artifact_axis) == 1:
                    isArti = 1
                elif max(artifact_axis) == 2:
                    if 1 in artifact_axis:
                        isArti = 1
                    else:
                        isArti = 2
                else:
                    isArti = 0
                    
                if self.case_num != None:
                    if isArti == 0 and self.amt_0 == 0:
                        pass
                    elif isArti != 0 and self.amt_1 == 0:
                        pass
                    else:
                        gtdict[key] = isArti

                        if isArti == 0:
                            self.amt_0 -= 1
                        else:
                            self.amt_1 -= 1
                else:
                    gtdict[key] = isArti
                
                
            except:
                pass
            
        return gtdict
    
    def min_max_scale(self, data):
        max_val = max(data)
        min_val = min(data)
        norm = [((float(val)-min_val)/(max_val-min_val)) for val in data]
        return norm
    
    def pad_cut(self, segment):
        if len(segment) == self.len_seg:
            pass
        elif len(segment) > self.len_seg:
            ctw = 0
            while (len(segment) != self.len_seg):
                if ctw == 0:
                    segment = np.delete(segment,0)
                    ctw = 1
                else:
                    segment = np.delete(segment,len(segment)-1)
                    ctw = 0
            #end of while loop
        elif len(segment) < self.len_seg:
            ctw = 0
            while (len(segment) != self.len_seg):
                if ctw == 0:
                    segment = np.insert(segment,0,0)
                    ctw = 1
                else:
                    segment = np.append(segment,0)
                    ctw = 0
            #end of while loop
        #end of pad/cut
        return segment
    
    def DA_Jitter(self,X, sigma=0.05):
        myNoise = np.random.normal(loc=0, scale=sigma, size=X.shape)
        return X+myNoise
    
    def DA_Cutout(self,X, cutlen=10):
        final_idx = random.randrange(len(X)-1)+10
        new_X = []
        for i in range(len(X)):
            if i > (final_idx - cutlen) and i < final_idx:
                new_X.append(min(X))
            else:
                new_X.append(X[i])
        #X[(final_idx-cutlen):final_idx] = np.zeros(cutlen)
        return np.asarray(new_X)

    def DA_Scaling(self,X, sigma=0.1):
        scalingFactor = np.random.normal(loc=1.0, scale=sigma, size=(X.shape[0])) # shape=(1,3)
        myNoise = np.matmul(np.ones((X.shape[0])), scalingFactor)
        return X*myNoise

    def GenerateRandomCurves(self,X, sigma=0.2, knot=4):
        xx = (np.ones((X.shape[0],1))*(np.arange(0,X.shape[0], (X.shape[0]-1)/(knot+1)))).transpose()
        yy = np.random.normal(loc=1.0, scale=sigma, size=(knot+2, X.shape[0]))
        x_range = np.arange(X.shape[0])
        cs_x = CubicSpline(xx[:,0], yy[:,0])
        #cs_y = CubicSpline(xx[:,1], yy[:,1])
        #cs_z = CubicSpline(xx[:,2], yy[:,2])
        return np.array([cs_x(x_range)]).transpose()


    def DA_MagWarp(self,X, sigma=0.2):
        return X.T * self.GenerateRandomCurves(X, sigma)

    def DistortTimesteps(self,X, sigma=0.2):
        tt = self.GenerateRandomCurves(X, sigma) # Regard these samples aroun 1 as time intervals
        tt_cum = np.cumsum(tt, axis=0)        # Add intervals to make a cumulative graph
        # Make the last value to have X.shape[0]
        t_scale = [(X.shape[0]-1)/tt_cum[-1,0]]
        tt_cum[:,0] = tt_cum[:,0]*t_scale[0]
        #tt_cum[:,1] = tt_cum[:,1]*t_scale[1]
        #tt_cum[:,2] = tt_cum[:,2]*t_scale[2]
        return tt_cum

    def DA_TimeWarp(self,X, sigma=0.2):
        tt_new = self.DistortTimesteps(X, sigma)
        X_new = np.zeros(X.shape)
        x_range = np.arange(X.shape[0])
        X_new = np.interp(x_range, tt_new.T[0], X)
        #X_new[:,1] = np.interp(x_range, tt_new[:,1], X[:,1])
        #X_new[:,2] = np.interp(x_range, tt_new[:,2], X[:,2])
        return X_new

    def DA_Permutation(self,X, nPerm=4, minSegLength=10):
        X_new = np.zeros(X.shape)
        idx = np.random.permutation(nPerm)
        bWhile = True
        while bWhile == True:
            segs = np.zeros(nPerm+1, dtype=int)
            segs[1:-1] = np.sort(np.random.randint(minSegLength, X.shape[0]-minSegLength, nPerm-1))
            segs[-1] = X.shape[0]
            if np.min(segs[1:]-segs[0:-1]) > minSegLength:
                bWhile = False
        pp = 0
        for ii in range(nPerm):
            x_temp = X[segs[idx[ii]]:segs[idx[ii]+1]]
            X_new[pp:pp+len(x_temp)] = x_temp
            pp += len(x_temp)
        return(X_new)

    def RandSampleTimesteps(self,X, nSample=1000):
        X_new = np.zeros(X.shape)
        tt = np.zeros((nSample,X.shape[1]), dtype=int)
        tt[1:-1,0] = np.sort(np.random.randint(1,X.shape[0]-1,nSample-2))
        tt[1:-1,1] = np.sort(np.random.randint(1,X.shape[0]-1,nSample-2))
        tt[1:-1,2] = np.sort(np.random.randint(1,X.shape[0]-1,nSample-2))
        tt[-1,:] = X.shape[0]-1
        return tt

    def change_df(self,df):
        time = []
        values = []
        increment = 1/75
        dp1 = df['dp1'].tolist()
        dp2 = df['dp2'].tolist()
        dp3 = df['dp3'].tolist()
        dp4 = df['dp4'].tolist()
        dp5 = df['dp5'].tolist()
        dp6 = df['dp6'].tolist()
        dp7 = df['dp7'].tolist()
        dp8 = df['dp8'].tolist()
        dp9 = df['dp9'].tolist()
        dp10 = df['dp10'].tolist()
        dp11 = df['dp11'].tolist()
        dp12 = df['dp12'].tolist()
        dp13 = df['dp13'].tolist()
        dp14 = df['dp14'].tolist()
        dp15 = df['dp15'].tolist()
        dp16 = df['dp16'].tolist()
        dp17 = df['dp17'].tolist()
        dp18 = df['dp18'].tolist()
        dp19 = df['dp19'].tolist()
        dp20 = df['dp20'].tolist()
        dp21 = df['dp21'].tolist()
        dp22 = df['dp22'].tolist()
        dp23 = df['dp23'].tolist()
        dp24 = df['dp24'].tolist()
        dp25 = df['dp25'].tolist()
        curr_time = 0
        for i in range(len(df)):
            for datapoint in [dp1,dp2,dp3,dp4,dp5,dp6,dp7,dp8,dp9,dp10,dp11,dp12,dp13,dp14,dp15,dp16,dp17,dp18,
                            dp19,dp20,dp21,dp22,dp23,dp24,dp25]:
                #values.append(datapoint[i])
                #time.append(curr_time)
                #curr_time = curr_time + increment
                try:
                    datapoint = float(datapoint[i])
                    values.append(datapoint)
                    time.append(curr_time)
                    curr_time = curr_time + increment
                except:
                    pass
    
        line_df = pd.DataFrame(
            {'value':values,
            'time':time
            })

        return line_df

    def get_cases(self,case_folder,cases_to_parse):
        all_sequences = {}
        dataset_path = case_folder
        for case in sorted(os.listdir(dataset_path)):
            if case != '.DS_Store' and case != '._.DS_Store' and case in self.cases_to_parse:
                if case[:2] == '._':
                    case = case[2:]
                    dict_key = str(case)
                for take in sorted(os.listdir(str(dataset_path+case))):
                    if take != '.DS_Store' and take != '._.DS_Store':
                        if take[:2] == '._':
                            take = take[2:]
                        dict_key = str(case+'_'+take)
                        for file in sorted(os.listdir(str(dataset_path+case+'/'+take))):
                            if (file != '.DS_Store' and file != '._.DS_Store') and file[6:11] == 'pleth':
                                pleth_data = pd.read_csv(dataset_path+case+'/'+take+'/'+file,names=['machID','dp1','dp2','dp3','dp4','dp5','dp6','dp7','dp8','dp9','dp10','dp11','dp12','dp13',
                                                                                                    'dp14','dp15','dp16','dp17','dp18','dp19','dp20','dp21','dp22','dp23','dp24','dp25','counter','date','take'])
                                counter = 0
                                member = file[12:16]
                                dict_key = str(case+'_'+take+'_'+member)
                                line_pleth = self.change_df(pleth_data)
                                for i in range(len(line_pleth)):
                                    all_sequences[dict_key] = line_pleth
                                    counter = 1
        return all_sequences

    def get_beat(self,all_sequences):
        
        #amt_p_case = int(len(self.cases_to_parse) / self.case_num)
        #end = False
        #ct = -1
        beat_sequences = {}
        for key in all_sequences:

            if self.end:
                break
            else:    
                key_count = 0
                seq = all_sequences[key]
                seq_val = seq['value'].to_numpy()
                peaks, _ = find_peaks(seq_val)
                onset, _ = find_peaks(-seq_val)
                peak_count = 0
                for i in range(len(onset)):
                            
                    key_name = key + '_'+ str(key_count)
                    if i == 0:
                        beat_sequences[key_name] = seq[0:onset[i]]
                        key_count = key_count + 1
                    else:
                        beat_sequences[key_name] = seq[onset[i-1]:onset[i]]
                        key_count = key_count + 1
        return beat_sequences
    
    def mount(self, dict_name):
        ct = dict_name.split('_')[3]
        #member = dict_name.split('_')[2]
        base_name = dict_name.split('_')[0] + '_' + dict_name.split('_')[1] + '_' + dict_name.split('_')[2] + '_'
        
        try:
            pleth_ctr = self.beats[dict_name]['value'].to_numpy()
        except:
            ctl = 1
            found = False
            while found == False:
                dict_name = base_name + str(int(ct)-ctl)
                if dict_name in self.beats:
                    found = True
                else:
                    ctl += 1
            pleth_ctr = self.beats[dict_name]['value'].to_numpy()
        
        #grab previous pleths
        pre_pleth = []
        for i in range(self.group_amt//2):
            new_dict_name = base_name + str(int(ct)-i)
            try:
                pre_pleth.append(self.beats[new_dict_name]['value'].to_numpy())
            except:
                pre_pleth.append(np.zeros(pleth_ctr.shape))
        #grab ahead pleths
        post_pleth = []
        for i in range(self.group_amt//2):
            new_dict_name = base_name + str(int(ct)+i)
            try:
                post_pleth.append(self.beats[new_dict_name]['value'].to_numpy())
            except:
                post_pleth.append(np.zeros(pleth_ctr.shape))
                
        # combine pre pleths
        cti = 0
        for i in reversed(range(len(pre_pleth))):
            if cti == 0:
                pleth = pre_pleth[i]
                cti += 1
            else:
                pleth_plus1 = pre_pleth[i]
                pleth = np.concatenate((pleth,pleth_plus1))

        #place target segment in the middle
        pleth = np.concatenate((pleth,pleth_ctr))

        # combine post pleths
        for i in range(len(post_pleth)):
            pleth_plus1 = post_pleth[i]
            pleth = np.concatenate((pleth,pleth_plus1))

        return pleth, pleth_ctr
         

    def __getitem__(self, idx):
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()
            
        if self.isLabel:
            dict_names = list(self.gtdict.keys())
        else:
            dict_names = list(self.beats.keys())
            
        dict_name = dict_names[idx]
        
        if self.isLabel:
            curr_gt = self.gtdict[dict_name]
        
        pleth, pleth_ctr = self.mount(dict_name)
        
        if self.isLabel:
            data_idx = self.min_max_scale(pleth)
                
            data_idx = self.pad_cut(data_idx)

            data_idx = np.asarray(data_idx)

            data_idx = np.nan_to_num(data_idx,nan=0.0,posinf=0.0,neginf=0.0)


            ### Substitute by TensorFlow "tensor"
            data_idx = np.float32(data_idx)

            if curr_gt == 2:
                curr_gt = 1

            sample = {'segment': data_idx, 'gt': curr_gt, 'pleth': pleth, 'pleth_ctr': pleth_ctr} 
        
        grabbed = False
        if self.isForecasting:
            ctf = 1
            while grabbed == False:
                try:
                    ct = dict_name.split('_')[3]
                    base_name = dict_name.split('_')[0] + '_' + dict_name.split('_')[1] + '_' + dict_name.split('_')[2] + '_'
                    new_dict_name = base_name + str(int(ct)+ctf)
                    grabbed = True
                except:
                    ctf += 1
            pleth_future, pleth_ctr = self.mount(new_dict_name)
            
            pleth = self.min_max_scale(pleth)
            pleth = self.pad_cut(pleth)
            pleth = np.asarray(pleth)
            pleth = np.nan_to_num(pleth,nan=0.0,posinf=0.0,neginf=0.0)

            ### Substitute by TensorFlow "tensor"
            pleth = np.float32(pleth)
            
            pleth_future = self.min_max_scale(pleth_future)
            pleth_future = self.pad_cut(pleth_future)
            pleth_future = np.asarray(pleth_future)
            pleth_future = np.nan_to_num(pleth_future,nan=0.0,posinf=0.0,neginf=0.0)

            ### Substitute by TensorFlow "tensor"
            pleth_future = np.float32(pleth_future)
            
            
            sample = {'x0': pleth, 'x1': pleth_future}
        
        if self.isFixmatch:
            data_idx = pleth
            
            ## Weak Augment
            rand_num = random.random()*100
            intensity = random.random()*10
            if rand_num < 33:
                data_idx_WA = self.DA_Jitter(data_idx,0.001*intensity)
            elif rand_num >= 33 and rand_num < 66:
                data_idx_WA = self.DA_Scaling(data_idx,0.5)
            else:
                data_idx_WA = self.DA_TimeWarp(data_idx,0.08*intensity)

            ## Strong Augment
            rand_num2 = random.random()*100
            intensity2 = random.random()*10
            if rand_num2 < 20:
                data_idx_SA = self.DA_Cutout(data_idx)
                data_idx_SA = self.DA_Jitter(data_idx_SA,0.001*intensity2)
                data_idx_SA = self.DA_Scaling(data_idx_SA,0.5)
            elif rand_num2 >= 20 and rand_num2 < 40:
                data_idx_SA = self.DA_Cutout(data_idx)
                data_idx_SA = self.DA_Jitter(data_idx_SA,0.001*intensity2)
                data_idx_SA = self.DA_Scaling(data_idx_SA,0.5)
                data_idx_SA = self.DA_TimeWarp(data_idx_SA,0.08*intensity2)
            elif rand_num2 >= 40 and rand_num2 < 60:
                data_idx_SA = self.DA_Cutout(data_idx)
                data_idx_SA = self.DA_Scaling(data_idx_SA,0.5)
                data_idx_SA = self.DA_TimeWarp(data_idx_SA,0.08*intensity2)
            elif rand_num2 >= 60 and rand_num2 < 80:
                data_idx_SA = self.DA_Jitter(data_idx,0.001*intensity2)
                data_idx_SA = self.DA_Scaling(data_idx_SA,0.5)
                data_idx_SA = self.DA_TimeWarp(data_idx_SA,0.08*intensity2)
            else:
                data_idx_SA = self.DA_Cutout(data_idx)
                data_idx_SA = self.DA_Jitter(data_idx_SA,0.001*intensity2)
                data_idx_SA = self.DA_TimeWarp(data_idx_SA,0.08*intensity2)


            data_idx_WA = self.min_max_scale(data_idx_WA)
            data_idx_SA = self.min_max_scale(data_idx_SA)
            data_idx = self.min_max_scale(data_idx)

            data_idx_WA = self.pad_cut(data_idx_WA)
            data_idx_SA = self.pad_cut(data_idx_SA)
            data_idx = self.pad_cut(data_idx)

            data_idx_SA = np.asarray(data_idx_SA)
            data_idx_WA = np.asarray(data_idx_WA)
            data_idx = np.asarray(data_idx)

            data_idx_SA = np.nan_to_num(data_idx_SA,nan=0.0,posinf=0.0,neginf=0.0)
            data_idx_WA = np.nan_to_num(data_idx_WA,nan=0.0,posinf=0.0,neginf=0.0)
            data_idx = np.nan_to_num(data_idx)
            


            ### Substitute by TensorFlow "tensor"
            data_idx_WA = np.float32(data_idx_WA)
            data_idx_SA = np.float32(data_idx_SA)
            data_idx = np.float32(data_idx)

            sample = {'WA': data_idx_WA, 'SA': data_idx_SA, 'NA': data_idx}
        
        return sample