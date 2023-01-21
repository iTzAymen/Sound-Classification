import math, random
import torch
from torch.utils.data import DataLoader, Dataset, random_split
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import torchaudio
from torchaudio import transforms
from IPython.display import Audio

from matplotlib import pyplot as plt

from os import walk
import pandas as pd

torch.Tensor.ndim = property(lambda self: len(self.shape)) #To allow plotting pytorch tensors

#Constants
MAX_AUDIO_LENGTH = 3000
SAMPLING_RATE = 48000
N_CHANNELS = 2

class AudioUtil():
    @staticmethod
    def open(audio_file):

        #Open an audio file
        # print(f"Opening file : {audio_file}")
        sig, sr = torchaudio.load(audio_file)
        return (sig, sr)
    
    @staticmethod
    def print(aud, channel):

        #Plot the audio signal wave

        sig, sr = aud
        duration = sig.shape[1]
        time = torch.linspace(0, duration/sr, duration)

        print(sig.shape)
        print('Plotting...')

        plt.figure(figsize=(15, 5))
        plt.plot(time, sig[channel - 1])
        plt.title('Audio Plot')
        plt.ylabel(' signal wave')
        plt.xlabel('time (s)')
        plt.show()

    @staticmethod
    def display_spectrogram(spec):
        
        #Display the audio mel spectrogram

        print(spec.shape)
        print('Plotting...')
        
        plt.imshow(spec[0])
        plt.title('Audio mel spectrogram')
        plt.ylabel('Frequency (mels)')
        plt.xlabel('Time (ms)')
        plt.colorbar(format='%+2.0f dB')

        plt.show()

    @staticmethod
    def rechannel(aud, new_channel):

        #Convert the audio from mono to stereo or vice versa

        sig, sr = aud

        if(sig.shape[0] == new_channel):
            return aud
        
        # print('Rechanneling to ' + str(new_channel))
        if(new_channel == 1):
            resig = sig[:1, :]
        else:
            resig = torch.cat([sig, sig])
        
        return ((resig, sr))

    @staticmethod
    def resample(aud, newsr):

        #Resample the audio to the newsr frequency

        sig, sr = aud

        if(sr == newsr):
            return
        
        # print('Resampling to ' + str(newsr))

        num_channels = sig.shape[0]
        resig = torchaudio.transforms.Resample(sr, newsr)(sig[:1, :])

        if(num_channels > 1):
            retwo = torchaudio.transforms.Resample(sr, newsr)(sig[1:, :])
            resig = torch.cat([resig, retwo])

        return((resig, newsr))

    @staticmethod
    def pad_trunc(aud, max_ms):

        #add padding, or truncate the signal to fit the max length
        sig, sr = aud
        num_rows, sig_len = sig.shape
        max_len = sr//1000 * max_ms

        if(sig_len > max_len):
            #Truncate the signal
            # print('Truncating signal to ' + str(max_ms) + ' ms')
            sig = sig[:, :max_len]
        elif(sig_len < max_len):
            #Add padding
            # print('Padding signal to ' + str(max_ms) + ' ms')
            pad_begin_len = random.randint(0, max_len - sig_len)
            pad_end_len = max_len - sig_len - pad_begin_len

            pad_begin = torch.zeros((num_rows, pad_begin_len))
            pad_end = torch.zeros((num_rows, pad_end_len))

            sig = torch.cat((pad_begin, sig, pad_end), 1)
        
        return ((sig, sr))

    @staticmethod
    def time_shift(aud, shift_limit):
        sig, sr = aud
        _, sig_len = sig.shape
        shift_amt = int(random.random() * shift_limit * sig_len)
        return (sig.roll(shift_amt), sr)

    @staticmethod
    def pitch_shift(aud, shift_limit):
        sig, sr = aud
        shift_amt = random.random() * shift_limit
        return (sig * shift_amt, sr)

    @staticmethod
    def get_mel_spectrogram(aud, hop_length):
        sig, sr = aud
        top_db = 80

        mel_transformation = torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_fft=1024, hop_length=hop_length, n_mels=64)
        db_transformation = torchaudio.transforms.AmplitudeToDB(top_db=top_db)
        spec = mel_transformation(sig)
        spec = db_transformation(spec)
        return spec

    @staticmethod
    def spectro_augment(spec, max_mask_pct = 0.1, n_freq_masks = 1, n_time_masks=1):
        _, n_mels, n_steps = spec.shape
        mask_value = spec.mean()
        aug_spec = spec

        freq_max_param = max_mask_pct * n_mels
        for _ in range(n_freq_masks):
            aug_spec = transforms.FrequencyMasking(freq_max_param)(aug_spec, mask_value)
        
        time_mask_params = max_mask_pct * n_steps
        for _ in range(n_time_masks):
            aug_spec = transforms.TimeMasking(time_mask_params)(aug_spec, mask_value)

        return aug_spec

    @staticmethod
    def preprocess_audio(audio_dir):
        aud = AudioUtil.open(audio_dir)
        aud = AudioUtil.rechannel(aud, N_CHANNELS)
        aud = AudioUtil.resample(aud, SAMPLING_RATE)
        aud = AudioUtil.pad_trunc(aud, MAX_AUDIO_LENGTH)
        aud = AudioUtil.time_shift(aud, 0.15)
        aud = AudioUtil.pitch_shift(aud, 1.25)
        spec = AudioUtil.get_mel_spectrogram(aud, 512)
        aug_spec = AudioUtil.spectro_augment(spec, n_freq_masks=2, n_time_masks=2)

        return (aud, spec, aug_spec)

class SoundDS(Dataset):
    def __init__(self, df, data_path):
        self.df = df
        self.data_path = data_path
        self.duration = MAX_AUDIO_LENGTH
        self.sr = SAMPLING_RATE
        self.channel = N_CHANNELS
        self.shift_pct = 0.15
    

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        audio_file = self.data_path + self.df.loc[index, 'relative_path']
        class_id = self.df.loc[index, 'classID']
        aud, spec, aug_spec = AudioUtil.preprocess_audio(audio_file)
        
        return aug_spec, class_id


d = {'relative_path' : [], 'classID': [], 'file_name': []}

def fillPaths(path, classID):
    for (dirpath, dirnames, filenames) in walk(path):
        relative_path = map(lambda self: dirpath + '/' + self, filenames)
        d['relative_path'].extend(relative_path)
        temp = [classID] * len(filenames)
        d['classID'].extend(temp)
        d['file_name'].extend(filenames)
        break

fillPaths('Dataset/Atraining_extrahls', 3)
fillPaths('Dataset/Atraining_murmur', 2)
fillPaths('Dataset/Atraining_normal', 1)
fillPaths('Dataset/Atraining_artifact', 0)

df = pd.DataFrame(data=d)
df = df[df.file_name != '.DS_Store']


myDS = SoundDS(df, '')

#Random split of 80:20 between training and validation
num_items = len(myDS)
num_train = int(num_items * 0.8)
num_val = num_items - num_train
# train_ds, val_ds = random_split(myDS, [num_train, num_val])
print(len(myDS))
path = df.iloc[97].relative_path
aud, spec, aug_spec = AudioUtil.preprocess_audio(path)
print(aud)
train_ds = []
val_ds = []
for i in range(num_train):
    _, _, spec = AudioUtil.preprocess_audio(df.iloc[i].relative_path)
    train_ds.append([spec, df.iloc[i].classID])
for i in range(num_train + 1, num_items):
    _, _, spec = AudioUtil.preprocess_audio(df.iloc[i].relative_path)
    val_ds.append([spec, df.iloc[i].classID])

#Create training and validation data loaders
train_dl = torch.utils.data.DataLoader(train_ds, batch_size = 16, shuffle = True)
val_dl = torch.utils.data.DataLoader(val_ds, batch_size = 16, shuffle = False)

# ----------------------------
# Audio Classification Model
# ----------------------------
class AudioClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        conv_layers = []

        #first conv block
        self.conv1 = nn.Conv2d(2, 8, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(8)
        init.kaiming_normal_(self.conv1.weight, a=0.1)
        self.conv1.bias.data.zero_()
        conv_layers += [self.conv1, self.relu1, self.bn1]

        #second conv block
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(16)
        init.kaiming_normal_(self.conv2.weight, a=0.1)
        self.conv2.bias.data.zero_()
        conv_layers += [self.conv2, self.relu2, self.bn2]

        #third conv block
        self.conv3 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2,2), padding=(1,1))
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(32)
        init.kaiming_normal_(self.conv3.weight, a=0.1)
        self.conv3.bias.data.zero_()
        conv_layers += [self.conv3, self.relu3, self.bn3]

        #fourth conv block
        self.conv4 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu4 = nn.ReLU()
        self.bn4 = nn.BatchNorm2d(64)
        init.kaiming_normal_(self.conv4.weight, a=0.1)
        conv_layers += [self.conv4, self.relu4, self.bn4]

        #linear classifier
        self.ap = nn.AdaptiveAvgPool2d(output_size=1)
        self.lin = nn.Linear(in_features=64, out_features=4)

        #wrap the convolutional blocks
        self.conv = nn.Sequential(*conv_layers)

    # ----------------------------
    # Forward pass computations
    # ----------------------------
    def forward(self, x):
        # Run the convolutional blocks
        x = self.conv(x)

        #Adaptive pool and flatten for input to linear layer
        x = self.ap(x)
        x = x.view(x.shape[0], -1)

        #Linear layer
        x = self.lin(x)

        #Final output
        return x
    
# Create the model and put it on gpu if available
myModel = AudioClassifier()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
myModel = myModel.to(device)
# Check that it is on cuda
next(myModel.parameters()).device

# ----------------------------
# Training Loop
# ----------------------------
def training(model, train_dl, num_epochs):
    # Loss function, Optimizer and Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001, steps_per_epoch=int(len(train_dl)), epochs=num_epochs, anneal_strategy='linear')

    # Repeat for each epoch
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_prediction = 0
        total_prediction = 0
        # Repeat for each batch in the training set
        for i, data in enumerate(train_dl):
            # Get the input features and target labels and put them on the GPU
            inputs, labels = data[0].to(device), data[1].to(device)
            # Normalize the inputs
            inputs_m, inputs_s = inputs.mean(), inputs.std()
            inputs = (inputs - inputs_m) / inputs_s

            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            #keep stats for loss and accuracy
            running_loss += loss.item()

            # get the predicted class with the highest score
            _, prediction = torch.max(outputs, 1)

            # count of predictions that matched the target label
            correct_prediction += (prediction == labels).sum().item()
            total_prediction += prediction.shape[0]
        
        # Print stats at the end of the epoch
        num_batches = len(train_dl)
        avg_loss = running_loss / num_batches
        acc = correct_prediction / total_prediction
        print(f'Epoch: {epoch}, Loss: {avg_loss:.2f}, Accuracy: {acc:.2f}')
    
    print('Finished Training')

num_epochs = 100
training(myModel, train_dl, num_epochs)