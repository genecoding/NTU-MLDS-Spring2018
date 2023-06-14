# Homework 2
* [2-1 Video Caption Generation]
* [2-2 Chatbot]

## Note
### 2-1 Video Caption Generation
* Implement [Sequence to Sequence - Video to Text (S2VT)]
* Training tips
  - [x] Use word embedding (instead of one-hot encoding)
  - [x] Attention (although the paper claims S2VT can work well without attention)
  - [x] Teacher forcing
  - [ ] Beam search
* Setup
  * Dataset  
    Select a caption for each video randomly (there are multiple captions for a video).
    ```python
    min_freq = 3
    ```
  * Other hyperparameters
    ```python
    VIDEO_FEAT_DIM = 4096
    INPUT_DIM = 200
    EN_HID_DIM = 1000
    DE_HID_DIM = 1000
    EMB_DIM = 200
    DROPOUT = 0.5
    batch_size = 128
    num_epochs = 200
    ```
### 2-2 Chatbot
* Use sequence-to-sequence model
* Training tips
  - [x] Use word embedding
  - [x] Attention
  - [x] Teacher forcing
  - [ ] Beam search (to do)
* Setup
  * Dataset  
    There are a lot of data (dialogue pairs) in this training set, to trim the data: use higher `min_freq`, constrain sentence length, remove pairs contain `<unk>` token(s); 
    also use RandomSampler to sample a fixed amount of data from the whole dataset randomly to shorten training time.
    ```python
    min_freq = 90
    MIN_LEN = 2
    MAX_LEN = 22
    NUM_TRAIN_SAMPLES = 100000
    NUM_VALID_SAMPLES = 5000
    ```
  * Other hyperparameters
    ```python
    num_layers = 2
    EMB_DIM = 1024
    EN_HID_DIM = 512
    DE_HID_DIM = 512
    DROPOUT = 0.5
    batch_size = 128
    num_epochs = 200    
    ```

## Result
Some decent results
### 2-1 Video Caption Generation
| video                | caption                        |
|----------------------|--------------------------------|
|![0lh_UWF9ZP4_62_69]  | A woman is mixing ingredients. |
|![7NNg0_n-bS8_21_30]  | A man is singing on a stage.   |
|![EpMuCrbxE8A_107_115]| A man is playing a guitar.     |
|![sJSmRik2c-c_1_7]    | A train is going down the road.|
|![TZ860P4iTaM_15_28]  | A cat is playing a piano.      |
### 2-2 Chatbot
```
input :	把 槍 放下 , 哈維
output: 哈維,放下槍放下

input :	哈維 , 放下 槍
output: 我求你哈維,哈維

input :	一個 殘酷 的 世界 裏 , 唯一 的 道德
output: 就是運氣

input :	我們 採取 了 行動
output: 我們三個並肩作戰

input :	那麼 失去 了 一切 的 人 , 爲什麼 就 我 一個 ?
output: 我不能忍受
```

## Reference
* https://vsubhashini.github.io/s2vt.html
* https://github.com/bentrevett/pytorch-seq2seq
* http://nlp.seas.harvard.edu/annotated-transformer/



[2-1 Video Caption Generation]: https://docs.google.com/presentation/d/1AeHW6-VDchIbjBXrOPQpXek82L3bi5PR5RapbOhcw94
[2-2 Chatbot]: https://docs.google.com/presentation/d/1GxaPl3_dGibYTlrg6WlvNTQHS2g4M30I37pVCZsS6tM
[Sequence to Sequence - Video to Text (S2VT)]: https://vsubhashini.github.io/s2vt.html
[0lh_UWF9ZP4_62_69]: 2-1/videos/0lh_UWF9ZP4_62_69.gif
[7NNg0_n-bS8_21_30]: 2-1/videos/7NNg0_n-bS8_21_30.gif
[EpMuCrbxE8A_107_115]: 2-1/videos/EpMuCrbxE8A_107_115.gif
[sJSmRik2c-c_1_7]: 2-1/videos/sJSmRik2c-c_1_7.gif
[TZ860P4iTaM_15_28]: 2-1/videos/TZ860P4iTaM_15_28.gif
