# HumanVoiceRecognition
# [人声/不同说话人声音]声纹检测
## ---基于循环神经网络
***
开源项目地址：https://github.com/pyannote/pyannote-audio
***
## 1.描述
pyannote.audio是一个用Python编写的开放源代码工具包，用于进行说话人检测。基于PyTorch深度学习框架，
提供了一组可训练的端到端神经构建块，可以将它们组合并共同优化以构建说话者差异化管道。
## 2.环境配置
官方推荐python3.7+，现所配置环境为ubuntu18.04 + python3.7.3，
在python3.6.9环境中安装成功，但未进行训练测试。
```shell
pip install pyannote-audio
```
因为官方推荐的数据集标注工具【Prodigy】需要收费,现使用开源数据集标工具label-studio，
label-studio项目地址:https://github.com/heartexlabs/label-studio。
```shell
pip install label-studio
```
使用深度学习框架pytorch1.6
## 2.本地数据集制作
### 2.1 音频文件准备
执行以下脚本以获取音频文件。
```python
import glob
import subprocess
import datetime


class WavExtractor():
    '''
    音频处理程序
    '''
    def __init__(self):
        self.videoFolder = '/home/winter/Video/*.mp4'  # 视频文件
        self.outputFolder = '/home/winter'  # 输出音频文件夹

    def extractor(self, video, wav):
        command = 'ffmpeg ' + '-i ' + video + ' -f ' + 'wav ' + wav
        subprocess.call(command, shell=True)

    def audioSave(self):
        todayDate = datetime.date.today().strftime('%m%d')
        videos = glob.glob(self.videoFolder)
        print(videos)
        for i in range(len(videos)):
            wavFile = self.outputFolder + '/' + 'audio_' + todayDate + '_' + str(i+1).zfill(2) + '.wav'
            try:
                self.extractor(videos[i], wavFile)
            except:
                pass


if __name__ == '__main__':
    w = WavExtractor().audioSave() ### 音频提取
    print("Done!")
```
### 2.2 配置文件
本文件中定义了两种标签："Speaking"和"Broadcast",
也可以继续定义多种标签，用以作多种声音分类。
需要配置config.xml文件，如下，将定义好的config.xml放入已经提取的音频文件夹中。
```xml
<View>
  <Labels name="label" toName="audio" choice="multiple">
    <Label value="Speaking" />
    <Label value="Broadcast" />
  </Labels>
  <AudioPlus name="audio" value="$audio"/>
</View>
```
### 2.3 音频文件标注
1. 为方便后续处理先将音频文件夹放入～/demo文件夹下并重命名为**amicorpus**;
2. 定义项目名为speakDetection,执行命令如下：
```shell
label-studio init --template=audio_diarization speakDetection --input-path=～/demo/amicorpus/ --input-format=audio-dir --label-config=～/demo/amicorpus/config.xml --allow-serving-local-files
label-studio start ./speakDetection
```
此过程会启动一个网页版的标注工具。
开始标注音频文件,标注方法详见:https://labelstud.io/playground/ 中Speaker diarization方法。
所有标注好的音频文件将会生成一个json文件，位于./skeakDetection/completions文件夹下。
(注：也可以将所有的标注结果导出到一个json文件或csv文件中，现只针对./completions单独json文件处理)
### 2.4 json文件处理
因为官方推荐的标注工具【Prodigy】收费，现使用label-studio标注的音频生成的json文件需要处理成特定的文件格式才能被模型加载；处理脚本较长，现只列出部分代码，只需定义好路径即可。
```python
class JsonLoader():
    def __init__(self):
        self.jsonFolder = '/home/winter/speaking/completions'  # json文件路径
        self.jsonFileList = self.jsonFolder + '/*.json'
        self.savedFilePath = '/home/winter/demo/AMI/'  # 处理好的生成文件保存路径
        self.trainPercent = 0.6  # 训练集比例
        self.testPercent = 0.2   # 测试集比例
        self.devPercent = 0.2    # 验证集比例
        self.allList = self.datasetSplit()
    ...
```
所有的生成文件将会位于**～/demo/AMI**路径下。
## 3.基于本地数据集模型训练
现定义根目录为～/demo
### 3.1 pyannote-studio源码下载
1. 需要将源码下载，使用里面有一些配置文件。
2. 将./pyannote-audio/tutorials/models/speech_activity_detection中的模型训练配置文**config.yml**件放入～/demo中。
3. 将./pyannote-audio/tutorials/data_preparation目录下的**MUSAN**文件夹和**database.yml**放入～/demo文件夹下；因使用自己的数据集，现需要修改database.yml中第二行内容如下：
```yml
##原第二行内容为AMI: ./amicorpus/*/audio/{uri}.wav
Databases:
   AMI: ./amicorpus/{uri}.wav
   MUSAN: ./musan/{uri}.wav

Protocols:
...
```
### 3.2 音频增强数据集下载
因为原论文中使用了音频数据增强的方法，现在需要下载相关的数据集，执行以下命令将MUSAN数据集下载并解压。
```shell
wget --continue -P ~/demo http://www.openslr.org/resources/17/musan.tar.gz
tar xzf ~/demo/musan.tar.gz -C ~/demo
```
demo文件夹内文件与文件夹：
- ～/demo
  - AMI 
  - amicorpus 
  - config.yml 
  - database.yml 
  - musan 
  - MUSAN 
### 3.3 模型训练
1. 定义所有文件夹根目录 
```shell
export EXP_DIR=~/demo
```
2. 声明数据集配置文件路径
```shell
export PYANNOTE_DATABASE_CONFIG=${EXP_DIR}/database.yml
```
3. 开始训练, sad为人声声纹识别模式训练;--to=100 为100个epoch；
```shell
pyannote-audio sad train --subset=train --to=100 --gpu --parallel=4 ${EXP_DIR} AMI.SpeakerDiarization.MixHeadset
```
scd可以用来作不同人声检测训练；
```shell
pyannote-audio scd train --subset=train --to=100 --gpu --parallel=4 ${EXP_DIR} AMI.SpeakerDiarization.MixHeadset
```
训练过程亦可参考：https://github.com/pyannote/pyannote-audio/tree/master/tutorials/models/speech_activity_detection
## 4.模型加载/验证模型
使用已经训练的模型进行验证
```shell
export TRN_DIR=${EXP_DIR}/train/AMI.SpeakerDiarization.MixHeadset.train
```
```shell
pyannote-audio sad validate --subset=development --from=10 --to=200 --every=10 ${TRN_DIR} AMI.SpeakerDiarization.MixHeadset
```
## 5.基于已给出的权重demo
作者已经给出一些训练好的权重:https://github.com/pyannote/pyannote-audio-hub
人声检测模型在./moudle/sad_ami.zip中
现给出人声检测测试代码:
```python
from pyannote.audio.features import Pretrained
from pyannote.audio.utils.signal import Binarize

sad = Pretrained(validate_dir='/home/winter/SpeechDetcet/sad_ami/train/AMI.SpeakerDiarization.MixHeadset.train/validate_detection_fscore/AMI.SpeakerDiarization.MixHeadset.development')

test_file = {'audio': '/home/winter/Resource/test_ruolin.wav'}
sad_scores = sad(test_file)

binarize = Binarize(offset=0.4, onset=0.95, log_scale=True, scale='percentile',
                    min_duration_off=0, min_duration_on=5)

speech = binarize.apply(sad_scores, dimension=1)
print(speech)
```
