from datetime import *
import pandas as pd
import numpy as np
import torch.utils.data as data
from tqdm import trange
store=pd.HDFStore('../data/ukdale/ukdale_h5')


def resample_meter_ukdale(store=None, building=1, meter=1, period='1min', cutoff=1000.):
    key = '/building{}/elec/meter{}'.format(building,meter)
    m = store[key]
    v = m.values.flatten()
    t = m.index
    s = pd.Series(v, index=t).clip(0.,cutoff)
    s[s<10.] = 0.
    return s.resample('1s').ffill(limit=300).fillna(0.).resample(period).mean().tz_convert('UTC')


def get_series_ukdale(datastore, house, label, cutoff):
    filename = '../data/ukdale/labels/house_%1d_labels.dat' %house
    #print(filename)
    labels = pd.read_csv(filename, delimiter=' ', header=None, index_col=0).to_dict()[1]
    
    for i in labels:
        if labels[i] == label:
            #print(i, labels[i])
            s = resample_meter_ukdale(store, house, i, '1min', cutoff)
            #s = resample_meter(store, house, i, '6s', cutoff)
    
    s.index.name = 'datetime'
    
    return s

def get_feather_train_valid_ukdale(store):
    houses=[1,2,5]
    ds=[]
    for house in houses:
        m = get_series_ukdale(store, house, 'aggregate', 10000.)
        m.name = 'aggregate'
        a1 = get_series_ukdale(store, house, 'kettle', 3100.)
        a1.name = 'kettle'
        if house==5:
            a2 = get_series_ukdale(store, house, 'fridge_freezer', 300.)
            a2.name = 'fridge'
        else:
            a2 = get_series_ukdale(store, house, 'fridge', 300.)
            a2.name = 'fridge'
        if house==5:
            a3 = get_series_ukdale(store, house, 'washer_dryer', 2500.)
            a3.name = 'washing_machine'
        else:
            a3 = get_series_ukdale(store, house, 'washing_machine', 2500.)
            a3.name = 'washing_machine'
        a4 = get_series_ukdale(store, house, 'microwave', 3000.)
        a4.name = 'microwave'
        if house==2:
            a5 = get_series_ukdale(store, house, 'dish_washer', 2500.)
            a5.name = 'dish_washer'
        else:
            a5 = get_series_ukdale(store, house, 'dishwasher', 2500.)
            a5.name = 'dish_washer'
        dsp= pd.concat([m, a1, a2, a3, a4, a5], axis=1)
        dsp.fillna(method='pad', inplace=True)
        ds.append(dsp)
    ds_1_train = ds[0][pd.datetime(2013,4,12):pd.datetime(2014,12,15)]
    ds_1_valid = ds[0][pd.datetime(2014,12,15):]    
    ds_2_train = ds[1][pd.datetime(2013,5,22):pd.datetime(2013,10,3,6,16)]
    ds_2_valid = ds[1][pd.datetime(2013,10,3,6,16):]
    ds_5_train = ds[2][pd.datetime(2014,6,29,tzinfo=timezone.utc):pd.datetime(2014,9,1,tzinfo=timezone.utc)]
    ds_5_valid = ds[2][pd.datetime(2014,9,1,tzinfo=timezone.utc):]
    
    ds_1_train.reset_index().to_feather('../data/ukdale/feather_files/UKDALE_1_train.feather')
    ds_2_train.reset_index().to_feather('../data/ukdale/feather_files/UKDALE_2_train.feather')
    ds_5_train.reset_index().to_feather('../data/ukdale/feather_files/UKDALE_5_train.feather')

    ds_1_valid.reset_index().to_feather('../data/ukdale/feather_files/UKDALE_1_valid.feather')
    ds_2_valid.reset_index().to_feather('../data/ukdale/feather_files/UKDALE_2_valid.feather')
    ds_5_valid.reset_index().to_feather('../data/ukdale/feather_files/UKDALE_5_valid.feather')
    
def get_status(app, threshold, min_off, min_on):
    condition = app > threshold
    # Find the indicies of changes in "condition"
    d = np.diff(condition)
    idx, = d.nonzero() 

    # We need to start things after the change in "condition". Therefore, 
    # we'll shift the index by 1 to the right.
    idx += 1

    if condition[0]:
        # If the start of condition is True prepend a 0
        idx = np.r_[0, idx]

    if condition[-1]:
        # If the end of condition is True, append the length of the array
        idx = np.r_[idx, condition.size] # Edit

    # Reshape the result into two columns
    idx.shape = (-1,2)
    on_events = idx[:,0].copy()
    off_events = idx[:,1].copy()
    assert len(on_events) == len(off_events)

    if len(on_events) > 0:
        off_duration = on_events[1:] - off_events[:-1]
        off_duration = np.insert(off_duration, 0, 1000.)
        on_events = on_events[off_duration > min_off]
        off_events = off_events[np.roll(off_duration, -1) > min_off]
        assert len(on_events) == len(off_events)

        on_duration = off_events - on_events
        on_events = on_events[on_duration > min_on]
        off_events = off_events[on_duration > min_on]

    s = app.copy()
    #s.iloc[:] = 0.
    s[:] = 0.

    for on, off in zip(on_events, off_events):
        #s.iloc[on:off] = 1.
        s[on:off] = 1.
    
    return s

class Power(data.Dataset):
    def __init__(self, meter=None, appliance=None, status=None, 
                 length=256, border=680, max_power=1., train=False):
        self.length = length
        self.border = border
        self.max_power = max_power
        self.train = train

        self.meter = meter.copy()/self.max_power
        self.appliance = appliance.copy()/self.max_power
        self.status = status.copy()

        self.epochs = (len(self.meter) - 2*self.border) // self.length
        
    def __getitem__(self, index):
        i = index * self.length + self.border
        if self.train:
            i = np.random.randint(self.border, len(self.meter) - self.length - self.border)

        x = self.meter.iloc[i-self.border:i+self.length+self.border].values.astype('float32')
        y = self.appliance.iloc[i:i+self.length].values.astype('float32')
        s = self.status.iloc[i:i+self.length].values.astype('float32')
        x -= x.mean()
        
        return x, y, s

    def __len__(self):
        return self.epochs
    

def chain2filter(aggregate, threshold, n_house):
    ds_1min = [aggregate[i].copy() for i in range(n_house)]
    for i in range(n_house):
        df = aggregate[i].copy()
        last_bin = 0
        for t in range(len(df)):
            if df.iloc[t] // threshold != last_bin:
                last_bin = df.iloc[t] // threshold
            else:
                df.iloc[t] = np.nan

        df[aggregate[i].resample('15min').first().index[1:]] = aggregate[i].resample('15min').first()[1:]

        aggregate[i] = df.reindex_like(aggregate[i]).ffill().fillna(0)
    return aggregate

def resample_meter_refit(store=None, Appliance=None, period='1min', cutoff=1000.):
    m = Appliance
    v = m.values.flatten()
    t = pd.to_datetime(store['Time'])
    s = pd.Series(v, index=t).clip(0.,cutoff)
    s[s<10.] = 0.
    return s.resample('1s').ffill(limit=300).fillna(0.).resample(period).mean()

def get_series_refit(house, label, cutoff):
    filename = '../data/refit/CLEAN_House%1d.csv' %house
    df = pd.read_csv(filename)
    labels=['Aggregate']
    for i in range(9):
        labels.append('Appliance%1d' %(i+1))
    for i in labels:
        if i == label:
            s = resample_meter_refit(df, df[i], '1min', cutoff)
            #s = resample_meter(store, house, i, '6s', cutoff)
    
    s.index.name = 'datetime'
    
    return s

def get_feather_refit():
    for house in trange(22):
        if house in [2,6]:
            m=get_series_refit(house, 'Aggregate', 10000.)
            m.name='aggregate'
            a1=get_series_refit(house,'Appliance3', 2500.)
            a1.name='dishwasher'
            globals()["ds_" + str(house)]=pd.concat([m,a1], axis=1)
            globals()["ds_" + str(house)].fillna(method='pad', inplace=True)
            globals()["ds_"+ str(house)].reset_index().to_feather('../data/refit/feather_files/dishwasher/REFIT_%d.feather' %house)
        if house in [5,9,11,13,15,21]:
            m=get_series_refit(house, 'Aggregate', 10000.)
            m.name='aggregate'
            a1=get_series_refit(house,'Appliance4', 2500.)
            a1.name='dishwasher'
            globals()["ds_" + str(house)]=pd.concat([m,a1], axis=1)
            globals()["ds_" + str(house)].fillna(method='pad', inplace=True)
            globals()["ds_"+ str(house)].reset_index().to_feather('../data/refit/feather_files/dishwasher/REFIT_%d.feather' %house)
        if house in [1,7,10,16,18]:
            m=get_series_refit(house, 'Aggregate', 10000.)
            m.name='aggregate'
            a1=get_series_refit(house,'Appliance6', 2500.)
            a1.name='dishwasher'
            globals()["ds_" + str(house)]=pd.concat([m,a1], axis=1)
            globals()["ds_" + str(house)].fillna(method='pad', inplace=True)
            globals()["ds_"+ str(house)].reset_index().to_feather('../data/refit/feather_files/dishwasher/REFIT_%d.feather' %house)
        if house in [3,20]:
            m=get_series_refit(house, 'Aggregate', 10000.)
            m.name='aggregate'
            a1=get_series_refit(house,'Appliance5', 2500.)
            a1.name='dishwasher'
            globals()["ds_" + str(house)]=pd.concat([m,a1], axis=1)
            globals()["ds_" + str(house)].fillna(method='pad', inplace=True)
            globals()["ds_"+ str(house)].reset_index().to_feather('../data/refit/feather_files/dishwasher/REFIT_%d.feather' %house)
        if house in [2,5,9,12,15]:
            m=get_series_refit(house, 'Aggregate', 10000.)
            m.name='aggregate'
            a1=get_series_refit(house,'Appliance1', 300.)
            a1.name='fridge'
            globals()["ds_" + str(house)]=pd.concat([m,a1], axis=1)
            globals()["ds_" + str(house)].fillna(method='pad', inplace=True)
            globals()["ds_"+ str(house)].reset_index().to_feather('../data/refit/feather_files/fridge/REFIT_%d.feather' %house)
        if house in [5,9,15]:
            m=get_series_refit(house, 'Aggregate', 10000.)
            m.name='aggregate'
            a1=get_series_refit(house,'Appliance3', 2500.)
            a1.name='washing_machine'
            globals()["ds_" + str(house)]=pd.concat([m,a1], axis=1)
            globals()["ds_" + str(house)].fillna(method='pad', inplace=True)
            globals()["ds_"+ str(house)].reset_index().to_feather('../data/refit/feather_files/washing_machine/REFIT_%d.feather' %house)
        if house in [7,16,18]:
            m=get_series_refit(house, 'Aggregate', 10000.)
            m.name='aggregate'
            a1=get_series_refit(house,'Appliance5', 2500.)
            a1.name='washing_machine'
            globals()["ds_" + str(house)]=pd.concat([m,a1], axis=1)
            globals()["ds_" + str(house)].fillna(method='pad', inplace=True)
            globals()["ds_"+ str(house)].reset_index().to_feather('../data/refit/feather_files/washing_machine/REFIT_%d.feather' %house)
        if house in [17,8]:
            m=get_series_refit(house, 'Aggregate', 10000.)
            m.name='aggregate'
            a1=get_series_refit(house,'Appliance4', 2500.)
            a1.name='washing_machine'
            globals()["ds_" + str(house)]=pd.concat([m,a1], axis=1)
            globals()["ds_" + str(house)].fillna(method='pad', inplace=True)
            globals()["ds_"+ str(house)].reset_index().to_feather('../data/refit/feather_files/washing_machine/REFIT_%d.feather' %house)
        if house==2:
            m=get_series_refit(house, 'Aggregate', 10000.)
            m.name='aggregate'
            a1=get_series_refit(house,'Appliance2', 2500.)
            a1.name='washing_machine'
            globals()["ds_" + str(house)]=pd.concat([m,a1], axis=1)
            globals()["ds_" + str(house)].fillna(method='pad', inplace=True)
            globals()["ds_"+ str(house)].reset_index().to_feather('../data/refit/feather_files/washing_machine/REFIT_%d.feather' %house)
        else:
            continue

    
           

            
        

        