'''
This script should be used to download and process x-ray images from the NIH 
database of chest x-rays. It goes all the way throug the steps of procesing the 
data in pytorch tensors. Afterwards other scripts should be used for training and testing
and experimentation with optimization alogorithms, kernel size, and transfer functions.
'''


######Read in Packages ##########
#Downloading and reading in images
import os
from glob import glob

#visualize images
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

#Preprocessing
import cv2
from skimage import io, transform
import numpy as np

#Transform,Dataloader for pytorch
from __future__ import print_function, division
import torch
import torchvision
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode


''' Using the OS package, we can use Python to write to the Linux command line. We'll use 
a wget command to download the NIH images right onto our VM instance. 'wget' is a tool to 
download whatever is at a specified URL like images, csv, json, etc. Here's the overall 
URL for the NIH dropbox folder of the x-ray images and metadata: https://nihcc.app.box.com/v/ChestXray-NIHCC

The tricky piece is getting the right URL for each image file. 
You might just be able to run the code below with success, but you will most likely have 
to click the download button for each image folder and pause the download. 
After, if you are using Chrome, go to your Downloads menu in the tool and you should 
see the link for the download. If Firefox, you should be able to copy the link address 
from the Download icon. Its the same process for the DataEntry.csv, which contains 
filenames and target labels for all of the images.  '''

#First set of images
os.system("wget -O images_001.tar.gz https://public.boxcloud.com/d/1/Rn0HLQmtcdcihU2LG0ETwRWpzvDA0mqFLBD4LQkeqeTzjHoiYd2F3aeK2mgg_i1do9q4NDjdwwxqwbZuZgawyktD_72vRKi1QC63dmynBqLNhCBOdh3TopMvQ-n5kCk5dTTr71QpNRpylGti_ohEdE1ZMPjnWsfVqbV1uLy3tYmJzw5zskKAjWKFMCB0jm9BM_ImTPThXuBoHvoUPJMg7mw2ulh3s2WrjhP2U7jPgY0Dw1hVNm8_Z-NZhwiwcg1Sa3HPcAtnhh3sXOTUuifbyZBGcB-jlGJ-oXCrNlKH6qVuVOuwl83FQaCxAEEjCVdCKhhgyq2Ve575a2sH00Zf6V9gEBjl2b_-M2Y0cxwwyVkvqktLzVXER7B4drAh4bAghtaKoEBo5Jo7Yy0mEpQcIX-1smrDXn2C0C3PPfD-4ANaQJCqxO3zShXxgcrsJoA-y5VwGari98pBgnUG_jap9Sy4xgIADLuMl6Gwzvr93qYgODgu8Lqh1XK-nscD3nXZudKETz4dPmD1rsDVAALcfjkIYTbalS2P0vxHOWQtyCra7dZv_eAlfDISgSBxvZhM419U3o5qHx4mdilnyds8--falqoKd8H-5Z14BKzDPrmUcGFmYFfnyoHrpoR6xD_OGj0rWYD_D-unlLF4ukLAXy1ahXtgBbZjKFbSoElCIGILqv00YRdWzjJqEtippQ0GrCaxVs2yUrAdJAqofSXsSDDNINop6hEsUGPuCxidjpgp2AoQdwGbkfKIzvxVvmWohK8n-qTTg2EKbXxS5QWlbS9D0ArojKGK04mTdP4n1fMJa3XtbKqU39PoaFm1OnU9a7fY_9rx5OITAx-XuJegeeAryoSYmy_7azdNBXpbmdn9UhpCaiQQfRDfHSkR371mnuVrFxgcKqSg7RqOeZKSEvDFNBAdoDe16PPiQ4AloJzKSobYBtzOc_7EOjy93mjxcGFHyTo71UxJNA9lNt9tAVK5BY2qxzxBoG-tsJsCRgTvM-yYxOv-ifgSlmL30A55-XEzXew_89VH3T53dGkmm_RyHSJUje6Taj3wRse-701F1J8qO32HKfFGRBFuegLRRLEuHOx20HjweDwyi2EP6_qSK8E5-Kvw_uZ4K51fD1Zfhz_idC2V-ouJtWaJDSZ71XdnYkQLioWvGF_QYhl6SHivNIrfU-m8M8CQTO-2oMFAvzLGAsUQQpAntGnNLLvMz_m0gvLkbd3P1U04JRH4Ao4kcL-IdWcYvg../download")
os.system("tar -xzvf images_001.tar.gz")


#download images002_tar.gz
os.system("wget -O images_002.tar.gz https://public.boxcloud.com/d/1/-QPmCKDbvw3vWciAb-eMAEAxw2Z4auMEVZBENRbdpCR6wASpHyhEoOHaJeTMHiNz91pC14uq1nDwbvnqJTlTdv6tZb7ufbddWZoBsaLvRo0yGHrSvEmpHtI9QA7WcGqXg-EQxSKYE5J_ddVJPPr2v_YxmOMeSrojlVdjVBSQ_bnlaDPAPS4PUrMfp_PQCu3DbyCCOinWoE3NPkpmG0Qyi1dt67w-DgqtlSmpxdun8q-lcPLpirRz41V4_kasFktLc7MVCfPr9I3_AIzJMYShCTHAmCB0kl0UShnTwCYZn9QQ1JyIJ38U4YhIgMktCl6sXWnBhcPtA6wLjDcDNeSATF1vdL9MOlfrGmCQH_f2d-TjpxRdKxMh-5LHkfwKubVBcF69Tosf9j6bfNPkpBtkaIhRkOzWoYhYXrRNX-DDseqf5ofaVQgeGSF83c3Eg546Wq2RBMWzPZz8E5KWVHGd_DxplP8SGqGvi6DwtODauHEiGONUPujHZE5XjXT5BY3TG9HNXVAxqmDua4Sv2p91cSYRcqbNn0qDfI4tWmlK7SuZH5SJM0CNOHhaCbvlelMEoafKPCgt1pRVRrlYGnGe6sipKUUkWBAxoG__6tQQRoFx2FelKa9G0vlbLNNK2eTfMOXwglmjW0x40OEWdwbgx8OK4i8Fy6Jv1vNr_9aM4MIQNEqIGEG98KZAejUKdhum0xj7rczap0ew6-Qn_onkiToBMeiC9C8_i1OdhMsdjGmNbkEdvpah8d45bNk1FBbanmRno4tTvC1CPayBidt4he17usm8PZrjvM0ktFdW7LVXyfJikFNta9TLAhu9ysASSWEUQcpwkHOhQU_O7sEJnRr-yCtA4PXF3sy7fH1vJ3QHD-jcHVCKyQPecTeJ1PMGyvNZ48hgYEr6BzfW6JQyfG4i7V6sHiRxAJNAvDq9g4iBiQbv18wkhDqqaUwRHO8I5CSv_4qBHNFftDN1Bc2R0rtrwa9hD6Pf7ZqrjsmgmPz43_sAly5KJLpKZOghqXOcgFQJc_5EEER3YR-vXBY5MLUZTZ0FDJGDPpTzPcqxdHJcWD18avyMBGhx2DUakVQ03Lw6tHkI4DW26lG_UXoRZeyPXum8ZOWHzUQef-UlvgPBcfhCk03JV5-i-a2Mg2ANq7OfMhGDZ1w2c9jyHwGR7e3FUJ43d8950HXaJHkdh15grs9g9ZKy4KZxlKlZa5YrFaeb_i5WN89JF0SY6RlnDy3mOiOXICmXsA../download")
os.system("tar -xzvf images_002.tar.gz")

#download images003_tar.gz
os.system("wget -O images_003.tar.gz https://public.boxcloud.com/d/1/NlJcFjkNSuuBU4c3IeFTn3cxz7EXrPp7i21GSQbfSaQZJIX02v8_oOS9IPUpIgT_2rFbT5kNFPl-kPvZNXIi1HZiWkHvg6JTDdLOHn81y45FKD-UyjY9ComUqcPORgh9jcfmVRJaaKF5527PzwwpNz09V3q5YGVmHEgPXShaQrTgJV0ASIfPb6eUh30OnFqLngGZXp8zNS1gNkBe0gtDtMXJo3QjXVB2yIn-pxJ6kqZeTbIuvc2EPQkLNcNoswVnAFbAhIyMt2GwLNYYHxo5ZjQ1_FMV_-cAKyi0n5CPgRw13XXCvVhcJgKDFnwjkXj_JOBxd-c631tNimdj3-ZdtQTRZbaheEc9a9XXMlsV5XAO1M5eTFfeSeHkU2vCmZMr90MuLhXaHTQGdXxMTvuRm2eDBAGCeRbDNn9C0ZU-jVNMtvETP1jSdMeRXENUd6PvE8JSSQWnUo3Oi42ndVVijCYj0OR1ZZBqxqX0ay3F8xuyrG-nX5y4NordA2MbzldBJbD2RfklgBrmv1ZoudF4-Em-rqfT92GPMtoGPV-aCdo2dfUpkN-DyBQhMrCxiPQ7ojSW9M1ZeNKkpTdMmUAvHoYd2m8pT22KaBi0n01RsxDGl8S0my80ZvYUuBb_rmlo389-IHk-FFvmFbHU-nQq9OlDW8TkdQ94nNkSs5UAKUkOu5fniaX0JrGKDyr0yDkCSFSYQrkygkyfinvvvKU0uf7TadMQJPJuJRmUt6jbut1HobQrBkDkzpg0Ml5tGVRKyGsu5hj7QQrummypCBJACDQpAJ3HCyWKgSMxNhDqfJqwRkPQKFNC4Zh6bIU-NYesEMA--urzT5InQIZt1-Nikixk3swyjD1bl-EzlF4KXVInllMpERgotTM5T3eOYFj3G-vio4UYXXDJeIwB8BR9qwvcpCL0gak6nTANJb88ieQeti5G0i4M4I1MDwLDRgAkogIuMOiujFNYzfDgiLRhrrW67k3vtsppiSEW7pveVkzimw5KTsz5vkmbm9mT8umus_rLLvpYG0UIXE0wDODNyTn8LAvtghDAKrGSgnlxMo_wOGyzrI6Ob5WP8uhFkJOSxICxiwoTe3to-zP99BS8saxzvCLpXnDfFl6mEVTA1TT5Xg4zq2HX0_6PBjQyu-3gUUuJXIVaW3xVugh4CbChDMwDKimIp8aisrx_Pefk24ZoCfuLpXcRLMUF2xH2bgiasBnrygNL_I6G8t-k8MZxXV__5JYulSS0NuFebhUPB0rYtJx8ww../download")
os.system("tar -xzvf images_003.tar.gz")


# ## Accessing Data in 'images' Folder of Working Directory

# Set a path, I printed what mine looks like where I called the working directory 'Final_Project'
path = os.path.abspath(os.path.join('..', 'Final_Project'))
print(path)


# Specify the location of the images  ../input/sample/images/ 
# - (When you call os.system('tar ...) above it should automatically create a images fodler)
source_images = os.path.join(path, "images")
print(source_images)


# #### Read in images using the 'glob' command

# Glob finds all the pathnames matching a specified pattern. 
#The  asterisk before .png is a regex type wildcard. It allows for anything to come 
#before .png so the glob looks for the pattern ../input/sample/images/*.png. # 
# We have to sort afterwards because it returns items in a random order, 
#and we want this index to match with dataentry.csv index. 
#Printing the output shows its in order and that between these three files we have 24,999 images.

imagesLocs = glob(os.path.join(source_images, "*.png"))
imagesLocs.sort()
print(len(imagesLocs))
print(imagesLocs[0:5])


# #### Read in CSV file form working directory
#### Uncomment below to download dataentry.csv #####
# os.system("wget -O dataentry.csv https://public.boxcloud.com/d/1/fn3IKt8NQfSLITGJJLAHVrn15KrD1cySF3iZil-_63dT_am7dYfZ-UBagEiblBk98kM_EbqQccSzdFvQP8Zup4ypnekIJowo-9cMDczYR2P52yi8LdmxBD5Lnfuyo7Gkbi0e8xqmY4u-UBScpq4JhRsT5GC57Yzr-btAVw4vNmuT7rLivELtzDQxjPGrAhZcnL71YcE86ac-OzcRApUmQTOjwiFYN3m7iCjNdo2Vxd5-rR08uxa1kiB7l--FDXugEJN6m5iU5ZFM5Eg738ufeed1uPZ4sAlFkotAnLuLwoYF0wOkxo2lUlXWwzNkMsKDigDr8eMTjLPKjZvb0CCHNvzdwy7T3ICKPqTKH-cP205R6FSjJ0me7-ezWhzCktXIul4W27CAgKhM9tNHGtNILj8wnhlRX6FQ87c82PuGH3q0gpa_MLR2tyeWdy6-0DQdHxHrXjyIMh3eSjhKbeSL89p-XFiCrRPDEjM73_EIP1wnOB7up2r1ELjh-lWo910WGuD-bRJizzHOBlP1rG6lfLRgbdoZze_4vJ7UP8h1p0mhj13m-rZwSPGLfKj7nK98AILD1r1Mn7jBV3XZYZqXyBA9EZu9n3bfubZcH_-4_64X002foNL7gD3v_J8bpZV3WvHAEsDobwhP33c7H_GmLjAqhMgylntZF1HByoKcbn6vSbFMCENjbYytFLIa2c-8n4KjcyTakSDY_4mn3vxWzVcIkd6-fIOQLcBB3yamnpn7j8hba8Hzvmvf-9WqxZMf5tjR7nD-tBTWW_hIAzvrwTr-c1IcwV1A2dCq30S6yetl8x4INGhtNbpz3zs9QFzOQeqsN2-wwOneuZMDrjn_9ShHVWhuEi640yQfurEVdHW_yNvQbhb59Lf2S6VcdpbM9tGejAsmZdYSPSOL-2UulLOKaI2eYqE0XcWtEfRbltbTNTndbC9pJyn_uKfMjHNdDBiS8VJTFVnUMjcdE7I3nK_qmcdSo79AxwNxxvDXvE-cJegogNP1V4rN2rjUDY3y0hKzRt5NH3gazWqlBJ2gdrLMg67WXhmVAAKCUUv8NHK4RyS_L8Q0FcbezSnPfNQUKkcl-ZWUQFJCk8YNrCJDeMjktgBUMQIpLLshzph7nrLMDkqTLJ0Z0-iXNuXM-02Ja-PJly0XK4dIARFn7oeOs1LDkGxMm5VEva74AVEF763mycru3eNfpUPgNDaDuRClXna3wbc./download")
labels = pd.read_csv('../Final_Project/dataentry.csv')
labels.head()


# ### Change Problem to Binary Classification #####
labels['Binary Label'] = labels.iloc[:,1] != 'No Finding' #create new binary label column
labels['Binary Label'] = labels["Binary Label"].astype(int) #represent as 1 for Finding and 0 for No Finding
labels = labels.drop(labels=['Finding Labels'], axis = 1) #Drop Finding Labels Column
labels = labels.iloc[:24999,:] #shorten

# Looking at the tail of the dataframe I see the last image index, and in the right most column are my 'Binary Label'
labels.tail()

# ### Visualize Raw Images
fig = plt.figure(figsize=(15,10))
for i in range(len(labels)):
    sample = plt.imread(imagesLocs[i])
        
    ax = plt.subplot(1,4,i+1)
    plt.tight_layout()
    ax.set_title('Sample #{}'.format(i))
    ax.axis('off')
    print(sample.shape, labels.loc[i,'Binary Label'])
    plt.imshow(sample)
    
    if i == 3:
        plt.show()
        break

# ### Resize and Save Images with CV2
imagesLocs = imagesLocs[:24999]
imagesLocs[-1]


# ### Resize, append, and save as 'x_images_arrays'

# cv2's resize method allows us to specify an image, a size to resize to 
# and an interpolation, which the method for pixel resampling in order to down scale the image.
# 
# Save to working directory using np.savez
x = []
for img in imagesLocs:
    fullImage = cv2.imread(img)
    x.append(cv2.resize(fullImage, (256,256), interpolation = cv2.INTER_AREA))
# x[0].shape
np.savez("x_images_arrays", x)


# ### Load Previously Resized Images and Visualize
# **Note**: If these are already in memory you can skip this, but run it in order to get your y labels

xImages = np.load('x_images_arrays.npz')['arr_0'] 
y  = labels.loc[:,'Binary Label'].values

# #### Plot Resized Images to Confirm Rescale
fig = plt.figure(figsize=(15,10))
for i in range(len(xImages)):
    sample = xImages[i]
        
    ax = plt.subplot(1,4,i+1)
    plt.tight_layout()
    ax.set_title('Sample #{}'.format(i))
    ax.axis('off')
    print(sample.shape, y[i], sample.max())
    plt.imshow(sample)
    
    if i == 3:
        plt.show()
        break


# ### Train Test Split

from sklearn.model_selection import train_test_split

# Create train, val, and test sets
X_train, X_dev, y_train, y_dev = train_test_split(xImages,y, shuffle =True, test_size = 0.2, random_state = 1)
X_test, X_val, y_test, y_val = train_test_split(X_dev, y_dev, shuffle =True, test_size = 0.2, random_state = 2)

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
print(X_val.shape, y_val.shape)

# ## At this point the data should be ready for us in whatever framework you choose to Use. 

#### PyTorch Specific Preprocessing ########
# #### Class to Apply Pytorch Dataset Wrapper to Images and Labels and Classes for Transform to Tensor and Random Crop

# Define Classes
class DataSetMod1(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, labels, images, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.labels = labels
                
        self.images = images
#         self.root_dir = root_dir
        self.transform = transform
        self.mean = np.mean(self.images)


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
#         img_name = os.path.join(self.root_dir,
#                                 self.labels.iloc[idx, 0])
        image = self.images[idx]
        image = (image)/255
        label = self.labels[idx]
        sample = {'image': image, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample
    
#Transformations:
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        
        return {'image': torch.from_numpy(image), 'label':label}
#Random Crop Class
class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image = sample['image']
        label = sample['label']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        return {'image': image, 'label': label}


# ### Wrap Train  in Dataset Wrapper with Random Crops to 224 and Convert to Tensor

transformedTrainData = DataSetMod1(labels=y_train, images = X_train, transform= transforms.Compose([RandomCrop(224), ToTensor()]))
transformedTestData = DataSetMod1(labels = y_test, images = X_test, transform= transforms.Compose([RandomCrop(224),ToTensor()]))
transformedValData = DataSetMod1(labels=y_val, images = X_val, transform = transforms.Compose([RandomCrop(224),ToTensor()]))

#Confirm that images are still looking normal
for i in range(len(transformedTrainData)):
    sample = transformedTrainData[i]
    plt.imshow(sample['image'].numpy().transpose(1, 2, 0))
    if i == 3:
        break

#### Instantiate Train, Test, and Val Loaders #######
trainDataLoader = DataLoader(transformedTrainData, batch_size=64,
                        shuffle=True, num_workers=0)
testDataLoader = DataLoader(transformedTestData, batch_size= 16, shuffle=False, num_workers=0)
valDataLoader = DataLoader(transformedValData)


# ### Define LeNet5