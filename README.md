# Pitch_difference
## Extract pitch and notes
(in ST_and_pitch folder)
```
pip install -r requirement.txt
```

```
python create_pitch_midi_pair.py [input mp3/mp4 file] [output json] -mp model\0425_70 -on 0.6 -off 0.5
```
Frame rate = 100 frames/sec  
  

## Generate dataset
```
python generate_dataset.py [output pkl]
```
  
![image](https://github.com/Jenny910927/Pitcg_difference/blob/main/Example_picture.png)
其中白色框框部分為training data、紅底線為validation data，將白色部分註解並取消註解紅色部分即可生成validation dataset。  
  
此dataset所含資料：
  * features
  * score_pitch (note)
  * pitch_diff (人聲實際音高與樂譜的音高差)
  * is_inlier (是否有差異過大的現象(主要是避免轉譜錯誤)，這裡設定pitch_diff>3者為False)
  * former_note (前一個note)
  * next_note (後一個note)
  * former_distance (此frame到note的起始點距離)
  * latter_distance (此frame到note的結束點距離)

## Training model
```
python train.py [input training dataset] [input validayion dataset]
```
  
`from predictor_1 import PitchDiffPredictor`  
這裡提供三種predictor：
  * predictor_1: loss只有使用L1 loss，計算實際與預測的差距
  * predictor_2: loss加上delta(相鄰frame之間預測出來的音高差距(一次微分))
  * predictor_3: loss加上delta和delta_2(delta的微分(二次微分))  
  
以及兩種net:
  * net_RNN_0807: 由2層RNN組成，中間用雙曲函數(torch.nn.functional.tanh)連接
  * net_RNN_position_embedding: score_pitch先經過position embedding，再train


## Dataset
提供原音檔(均由Youtube下載)與轉譜後的json檔，以及組成dataset的Pickle檔  
其中train.pkl由19首歌組成、val.pkl / val_2.pkl各由1首歌組成(歌手均為日本唱見そらる(soraru)演唱)
https://drive.google.com/drive/folders/1CRuiremCjD-T6WuluLnAmplHWAWggywz?usp=sharing  
