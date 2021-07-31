# chinese_qa_with_bert

## Model


## Get Started pretrain 


https://miro.medium.com/max/875/1*EZys07JxDTqJvyovZIetnA.png

![radar](https://miro.medium.com/max/875/1*EZys07JxDTqJvyovZIetnA.png)



## Test model

```
python -m interactive --output_dir output --model_type bert  --predict_file input.json --state_dict output/pytorch_model.bin --model_name_or_path bert-base-chinese
```

```
Please Enter:{"context": "江苏路街道是中国上海市长宁区下辖的一个街道办事处，位于长宁区最东部，东到镇宁路接邻静安区的静安寺街道，北到武定西路接邻静安区的曹家渡 街道，南到华山路接邻徐汇区的湖南路街道，西部界限为长宁路、安西路、武夷路等。面积1.52平方公里，户籍人口5.26万人，下辖13个居委会。长宁区区政府设在该街道辖区内的 愚园路（近安西路）。江苏路街道的主要街道江苏路、愚园路、长宁路、武夷路、武定西路、延安西路，均为上海公共租界越界筑路，是花园洋房集中的街区，是愚园路历史文化风 貌区的主体部分。较著名的近代建筑有中西女中（今上海市第三女子中学）、王伯群及汪精卫公馆（今长宁区少年宫）、西园公寓等。江苏路、长宁路、延安西路（高架）等经过拓 宽，已经形成上海市的交通干道。上海市轨道交通二号线经过该街道辖区，设有江苏路站。江苏路街道下辖歧山居委会、江苏居委会、万村居委会、南汪居委会、东浜居委会、愚三 居委会、曹家堰居委会、西浜居委会、福世居委会、长新居委会、华山居委会、利西居委会、北汪居委会。","qas": [{"question": "江苏路街道在上海市的什么地方？"}]}
Evaluating: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  2.61it/s]

江 苏 路 街 道 是 中 国 上 海 市 长 
```

輸入文本與問題

資料預處理

序列資料:
滑動窗格並以固定上限字數(max_seq_length)自動切割文本，整合為序列資料
每一序列為固定上限字數，最後未達固定上限字數之序列資料以補0表示


### 模型輸入

input_ids = 文字轉id [CLS]+Q+[SEP]+文本+[SEP]

input_mask = 有字的位置為1, 否則為0 [CLS]+Q+[SEP]+文本+[SEP] 

segment_ids = 問題為0, 文本為1 [CLS]+Q+[SEP]=0  文本+[SEP]=1 #沒用到(?)

start_position = 答案在[CLS]+Q+[SEP]+文本[SEP] 的開始位置 若沒有答案 訓練為0 測試為None

end_position = 答案在[CLS]+Q+[SEP]+文本[SEP] 的結束位置 若沒有答案 訓練為0 測試為None
```
input_ids= [101, 3736, ....., 833, 102]   
input_mask= [1, ...., 1] 
segment_ids= [0, 0, 0,  0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] 
start_position= None 
end_position= None
```

### 資料前處理

序列資料:

滑動窗格並以固定上限字數(max_seq_length)自動切割文本，整合為序列資料

每一序列為固定上限字數，最後未達固定上限字數之序列資料以補0表示

![radar]https://myppt.cc/lypGC7

![radar]https://ppt.cc/f9VRTx@.png

例如:

max_seq_length文本長度限制:384

max_tokens_for_doc一次序列文本限制:max_seq_length-問題(15)-3=366

len(all_doc_tokens)文本長度 = 434 

max_query_length問題長度限制= 64

len(query_tokens)問題長度 = 15 

文本可以分為兩個序列(滑動窗格=128)
```
_DocSpan(start=start_offset, length=length)=
DocSpan(start=0, length=366)
_DocSpan(start=start_offset, length=length)=
DocSpan(start=128, length=306)
```
答案開頭與結尾為未知
```
============doc_span_index 0 ============
input_ids= [101, 3736, 5722, 6662, 6125, 6887, 1762, 677, 3862, 2356, 4638, 784, 720, 1765, 3175, 8043, 102, 3736, 5722, 6662, 6125, 6887, 3221, 704, 1744, 677, 3862, 2356, 7270, 2123, 1277, 678, 6785, 4638, 671, 702, 6125, 6887, 1215, 752, 1905, 8024, 855, 754, 7270, 2123, 1277, 3297, 691, 6956, 8024, 691, 1168, 7252, 2123, 6662, 2970, 6943, 7474, 2128, 1277, 4638, 7474, 2128, 2191, 6125, 6887, 8024, 1266, 1168, 3636, 2137, 6205, 6662, 2970, 6943, 7474, 2128, 1277, 4638, 3293, 2157, 3941, 6125, 6887, 8024, 1298, 1168, 1290, 2255, 6662, 2970, 6943, 2528, 3726, 1277, 4638, 3959, 1298, 6662, 6125, 6887, 8024, 6205, 6956, 4518, 7361, 711, 7270, 2123, 6662, 510, 2128, 6205, 6662, 510, 3636, 1929, 6662, 5023, 511, 7481, 4916, 122, 119, 126, 123, 2398, 3175, 1062, 7027, 8024, 2787, 5093, 782, 1366, 126, 119, 123, 127, 674, 782, 8024, 678, 6785, 122, 124, 702, 2233, 1999, 833, 511, 7270, 2123, 1277, 1277, 3124, 2424, 6392, 1762, 6421, 6125, 6887, 6785, 1277, 1079, 4638, 2694, 1736, 6662, 8020, 6818, 2128, 6205, 6662, 8021, 511, 3736, 5722, 6662, 6125, 6887, 4638, 712, 6206, 6125, 6887, 3736, 5722, 6662, 510, 2694, 1736, 6662, 510, 7270, 2123, 6662, 510, 3636, 1929, 6662, 510, 3636, 2137, 6205, 6662, 510, 2454, 2128, 6205, 6662, 8024, 1772, 711, 677, 3862, 1062, 1066, 4909, 4518, 6632, 4518, 5029, 6662, 8024, 3221, 5709, 1736, 3817, 2791, 7415, 704, 4638, 6125, 1277, 8024, 3221, 2694, 1736, 6662, 1325, 1380, 3152, 1265, 7599, 6505, 1277, 4638, 712, 860, 6956, 1146, 511, 6772, 5865, 1399, 4638, 6818, 807, 2456, 5029, 3300, 704, 6205, 1957, 704, 8020, 791, 677, 3862, 2356, 5018, 676, 1957, 2094, 704, 2110, 8021, 510, 4374, 843, 5408, 1350, 3742, 5125, 1310, 1062, 7667, 8020, 791, 7270, 2123, 1277, 2208, 2399, 2151, 8021, 510, 6205, 1736, 1062, 2171, 5023, 511, 3736, 5722, 6662, 510, 7270, 2123, 6662, 510, 2454, 2128, 6205, 6662, 8020, 7770, 3373, 8021, 5023, 5307, 6814, 2868, 2160, 8024, 2347, 5307, 2501, 2768, 677, 3862, 2356, 4638, 769, 6858, 2397, 6887, 511, 677, 3862, 2356, 6758, 6887, 769, 6858, 753, 1384, 5296, 5307, 6814, 6421, 6125, 6887, 6785, 1277, 8024, 6392, 3300, 3736, 5722, 6662, 4991, 511, 3736, 5722, 6662, 6125, 6887, 678, 6785, 3637, 2255, 2233, 1999, 833, 510, 3736, 5722, 2233, 1999, 833, 102]
input_mask= [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
segment_ids= [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
start_position= None
end_position= None
============doc_span_index 1 ============
input_ids= [101, 3736, 5722, 6662, 6125, 6887, 1762, 677, 3862, 2356, 4638, 784, 720, 1765, 3175, 8043, 102, 122, 124, 702, 2233, 1999, 833, 511, 7270, 2123, 1277, 1277, 3124, 2424, 6392, 1762, 6421, 6125, 6887, 6785, 1277, 1079, 4638, 2694, 1736, 6662, 8020, 6818, 2128, 6205, 6662, 8021, 511, 3736, 5722, 6662, 6125, 6887, 4638, 712, 6206, 6125, 6887, 3736, 5722, 6662, 510, 2694, 1736, 6662, 510, 7270, 2123, 6662, 510, 3636, 1929, 6662, 510, 3636, 2137, 6205, 6662, 510, 2454, 2128, 6205, 6662, 8024, 1772, 711, 677, 3862, 1062, 1066, 4909, 4518, 6632, 4518, 5029, 6662, 8024, 3221, 5709, 1736, 3817, 2791, 7415, 704, 4638, 6125, 1277, 8024, 3221, 2694, 1736, 6662, 1325, 1380, 3152, 1265, 7599, 6505, 1277, 4638, 712, 860, 6956, 1146, 511, 6772, 5865, 1399, 4638, 6818, 807, 2456, 5029, 3300, 704, 6205, 1957, 704, 8020, 791, 677, 3862, 2356, 5018, 676, 1957, 2094, 704, 2110, 8021, 510, 4374, 843, 5408, 1350, 3742, 5125, 1310, 1062, 7667, 8020, 791, 7270, 2123, 1277, 2208, 2399, 2151, 8021, 510, 6205, 1736, 1062, 2171, 5023, 511, 3736, 5722, 6662, 510, 7270, 2123, 6662, 510, 2454, 2128, 6205, 6662, 8020, 7770, 3373, 8021, 5023, 5307, 6814, 2868, 2160, 8024, 2347, 5307, 2501, 2768, 677, 3862, 2356, 4638, 769, 6858, 2397, 6887, 511, 677, 3862, 2356, 6758, 6887, 769, 6858, 753, 1384, 5296, 5307, 6814, 6421, 6125, 6887, 6785, 1277, 8024, 6392, 3300, 3736, 5722, 6662, 4991, 511, 3736, 5722, 6662, 6125, 6887, 678, 6785, 3637, 2255, 2233, 1999, 833, 510, 3736, 5722, 2233, 1999, 833, 510, 674, 3333, 2233, 1999, 833, 510, 1298, 3742, 2233, 1999, 833, 510, 691, 3853, 2233, 1999, 833, 510, 2694, 676, 2233, 1999, 833, 510, 3293, 2157, 1840, 2233, 1999, 833, 510, 6205, 3853, 2233, 1999, 833, 510, 4886, 686, 2233, 1999, 833, 510, 7270, 3173, 2233, 1999, 833, 510, 1290, 2255, 2233, 1999, 833, 510, 1164, 6205, 2233, 1999, 833, 510, 1266, 3742, 2233, 1999, 833, 511, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
input_mask= [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
segment_ids= [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
start_position= None
end_position= None
```
### 模型架構

![radar](https://ppt.cc/fTRanx@.png)

模型有12層(假設有6個滑動窗格)

經過 BertEncoder、(BertLayer、BertAttention、BertSelfAttention)*12
```
====BertEncoder====
=====i- 0 =====
hidden_states= torch.Size([6, 384, 768])
attention_mask= torch.Size([6, 1, 1, 384])
layer_head_mask= None
encoder_hidden_states= None
encoder_attention_mask= None
past_key_value= None
output_attentions= False
==BertLayer===
===BertAttention===
===BertSelfAttention===
mixed_query_layer= torch.Size([6, 384, 768])
query_layer= torch.Size([6, 12, 384, 64])
query_layer= torch.Size([6, 12, 384, 64])
query_layer= torch.Size([6, 12, 384, 64])
attention_scores.size()= torch.Size([6, 12, 384, 384])
attention_probs.size()= torch.Size([6, 12, 384, 384])
attention_probs.size()= torch.Size([6, 12, 384, 384])
context_layer.size()= torch.Size([6, 12, 384, 64])
context_layer.size()= torch.Size([6, 384, 12, 64])
context_layer.size()= torch.Size([6, 384, 768])
attention_output= torch.Size([6, 384, 768])
attention_output.size()= torch.Size([6, 384, 768])
layer_output= torch.Size([6, 384, 768])
hidden_states= torch.Size([6, 384, 768])
layer_outputs= <class 'tuple'>
len(layer_outputs)= 1
type(layer_outputs)= <class 'tuple'>
i= <class 'torch.Tensor'> len= 6
i size()= torch.Size([6, 384, 768])
==========
=====i- 1 =====
hidden_states= torch.Size([6, 384, 768])
attention_mask= torch.Size([6, 1, 1, 384])
layer_head_mask= None
encoder_hidden_states= None
encoder_attention_mask= None
past_key_value= None
output_attentions= False
==BertLayer===
===BertAttention===
===BertSelfAttention===
mixed_query_layer= torch.Size([6, 384, 768])
query_layer= torch.Size([6, 12, 384, 64])
query_layer= torch.Size([6, 12, 384, 64])
query_layer= torch.Size([6, 12, 384, 64])
attention_scores.size()= torch.Size([6, 12, 384, 384])
attention_probs.size()= torch.Size([6, 12, 384, 384])
attention_probs.size()= torch.Size([6, 12, 384, 384])
context_layer.size()= torch.Size([6, 12, 384, 64])
context_layer.size()= torch.Size([6, 384, 12, 64])
context_layer.size()= torch.Size([6, 384, 768])
attention_output= torch.Size([6, 384, 768])
attention_output.size()= torch.Size([6, 384, 768])
layer_output= torch.Size([6, 384, 768])
hidden_states= torch.Size([6, 384, 768])
layer_outputs= <class 'tuple'>
len(layer_outputs)= 1
type(layer_outputs)= <class 'tuple'>
i= <class 'torch.Tensor'> len= 6
i size()= torch.Size([6, 384, 768])
==========
=====i- 2 =====
hidden_states= torch.Size([6, 384, 768])
attention_mask= torch.Size([6, 1, 1, 384])
layer_head_mask= None
encoder_hidden_states= None
encoder_attention_mask= None
past_key_value= None
output_attentions= False
==BertLayer===
===BertAttention===
===BertSelfAttention===
mixed_query_layer= torch.Size([6, 384, 768])
query_layer= torch.Size([6, 12, 384, 64])
query_layer= torch.Size([6, 12, 384, 64])
query_layer= torch.Size([6, 12, 384, 64])
attention_scores.size()= torch.Size([6, 12, 384, 384])
attention_probs.size()= torch.Size([6, 12, 384, 384])
attention_probs.size()= torch.Size([6, 12, 384, 384])
context_layer.size()= torch.Size([6, 12, 384, 64])
context_layer.size()= torch.Size([6, 384, 12, 64])
context_layer.size()= torch.Size([6, 384, 768])
attention_output= torch.Size([6, 384, 768])
attention_output.size()= torch.Size([6, 384, 768])
layer_output= torch.Size([6, 384, 768])
hidden_states= torch.Size([6, 384, 768])
layer_outputs= <class 'tuple'>
len(layer_outputs)= 1
type(layer_outputs)= <class 'tuple'>
i= <class 'torch.Tensor'> len= 6
i size()= torch.Size([6, 384, 768])
==========
=====i- 3 =====
hidden_states= torch.Size([6, 384, 768])
attention_mask= torch.Size([6, 1, 1, 384])
layer_head_mask= None
encoder_hidden_states= None
encoder_attention_mask= None
past_key_value= None
output_attentions= False
==BertLayer===
===BertAttention===
===BertSelfAttention===
mixed_query_layer= torch.Size([6, 384, 768])
query_layer= torch.Size([6, 12, 384, 64])
query_layer= torch.Size([6, 12, 384, 64])
query_layer= torch.Size([6, 12, 384, 64])
attention_scores.size()= torch.Size([6, 12, 384, 384])
attention_probs.size()= torch.Size([6, 12, 384, 384])
attention_probs.size()= torch.Size([6, 12, 384, 384])
context_layer.size()= torch.Size([6, 12, 384, 64])
context_layer.size()= torch.Size([6, 384, 12, 64])
context_layer.size()= torch.Size([6, 384, 768])
attention_output= torch.Size([6, 384, 768])
attention_output.size()= torch.Size([6, 384, 768])
layer_output= torch.Size([6, 384, 768])
hidden_states= torch.Size([6, 384, 768])
layer_outputs= <class 'tuple'>
len(layer_outputs)= 1
type(layer_outputs)= <class 'tuple'>
i= <class 'torch.Tensor'> len= 6
i size()= torch.Size([6, 384, 768])
==========
=====i- 4 =====
hidden_states= torch.Size([6, 384, 768])
attention_mask= torch.Size([6, 1, 1, 384])
layer_head_mask= None
encoder_hidden_states= None
encoder_attention_mask= None
past_key_value= None
output_attentions= False
==BertLayer===
===BertAttention===
===BertSelfAttention===
mixed_query_layer= torch.Size([6, 384, 768])
query_layer= torch.Size([6, 12, 384, 64])
query_layer= torch.Size([6, 12, 384, 64])
query_layer= torch.Size([6, 12, 384, 64])
attention_scores.size()= torch.Size([6, 12, 384, 384])
attention_probs.size()= torch.Size([6, 12, 384, 384])
attention_probs.size()= torch.Size([6, 12, 384, 384])
context_layer.size()= torch.Size([6, 12, 384, 64])
context_layer.size()= torch.Size([6, 384, 12, 64])
context_layer.size()= torch.Size([6, 384, 768])
attention_output= torch.Size([6, 384, 768])
attention_output.size()= torch.Size([6, 384, 768])
layer_output= torch.Size([6, 384, 768])
hidden_states= torch.Size([6, 384, 768])
layer_outputs= <class 'tuple'>
len(layer_outputs)= 1
type(layer_outputs)= <class 'tuple'>
i= <class 'torch.Tensor'> len= 6
i size()= torch.Size([6, 384, 768])
==========
=====i- 5 =====
hidden_states= torch.Size([6, 384, 768])
attention_mask= torch.Size([6, 1, 1, 384])
layer_head_mask= None
encoder_hidden_states= None
encoder_attention_mask= None
past_key_value= None
output_attentions= False
==BertLayer===
===BertAttention===
===BertSelfAttention===
mixed_query_layer= torch.Size([6, 384, 768])
query_layer= torch.Size([6, 12, 384, 64])
query_layer= torch.Size([6, 12, 384, 64])
query_layer= torch.Size([6, 12, 384, 64])
attention_scores.size()= torch.Size([6, 12, 384, 384])
attention_probs.size()= torch.Size([6, 12, 384, 384])
attention_probs.size()= torch.Size([6, 12, 384, 384])
context_layer.size()= torch.Size([6, 12, 384, 64])
context_layer.size()= torch.Size([6, 384, 12, 64])
context_layer.size()= torch.Size([6, 384, 768])
attention_output= torch.Size([6, 384, 768])
attention_output.size()= torch.Size([6, 384, 768])
layer_output= torch.Size([6, 384, 768])
hidden_states= torch.Size([6, 384, 768])
layer_outputs= <class 'tuple'>
len(layer_outputs)= 1
type(layer_outputs)= <class 'tuple'>
i= <class 'torch.Tensor'> len= 6
i size()= torch.Size([6, 384, 768])
==========
=====i- 6 =====
hidden_states= torch.Size([6, 384, 768])
attention_mask= torch.Size([6, 1, 1, 384])
layer_head_mask= None
encoder_hidden_states= None
encoder_attention_mask= None
past_key_value= None
output_attentions= False
==BertLayer===
===BertAttention===
===BertSelfAttention===
mixed_query_layer= torch.Size([6, 384, 768])
query_layer= torch.Size([6, 12, 384, 64])
query_layer= torch.Size([6, 12, 384, 64])
query_layer= torch.Size([6, 12, 384, 64])
attention_scores.size()= torch.Size([6, 12, 384, 384])
attention_probs.size()= torch.Size([6, 12, 384, 384])
attention_probs.size()= torch.Size([6, 12, 384, 384])
context_layer.size()= torch.Size([6, 12, 384, 64])
context_layer.size()= torch.Size([6, 384, 12, 64])
context_layer.size()= torch.Size([6, 384, 768])
attention_output= torch.Size([6, 384, 768])
attention_output.size()= torch.Size([6, 384, 768])
layer_output= torch.Size([6, 384, 768])
hidden_states= torch.Size([6, 384, 768])
layer_outputs= <class 'tuple'>
len(layer_outputs)= 1
type(layer_outputs)= <class 'tuple'>
i= <class 'torch.Tensor'> len= 6
i size()= torch.Size([6, 384, 768])
==========
=====i- 7 =====
hidden_states= torch.Size([6, 384, 768])
attention_mask= torch.Size([6, 1, 1, 384])
layer_head_mask= None
encoder_hidden_states= None
encoder_attention_mask= None
past_key_value= None
output_attentions= False
==BertLayer===
===BertAttention===
===BertSelfAttention===
mixed_query_layer= torch.Size([6, 384, 768])
query_layer= torch.Size([6, 12, 384, 64])
query_layer= torch.Size([6, 12, 384, 64])
query_layer= torch.Size([6, 12, 384, 64])
attention_scores.size()= torch.Size([6, 12, 384, 384])
attention_probs.size()= torch.Size([6, 12, 384, 384])
attention_probs.size()= torch.Size([6, 12, 384, 384])
context_layer.size()= torch.Size([6, 12, 384, 64])
context_layer.size()= torch.Size([6, 384, 12, 64])
context_layer.size()= torch.Size([6, 384, 768])
attention_output= torch.Size([6, 384, 768])
attention_output.size()= torch.Size([6, 384, 768])
layer_output= torch.Size([6, 384, 768])
hidden_states= torch.Size([6, 384, 768])
layer_outputs= <class 'tuple'>
len(layer_outputs)= 1
type(layer_outputs)= <class 'tuple'>
i= <class 'torch.Tensor'> len= 6
i size()= torch.Size([6, 384, 768])
==========
=====i- 8 =====
hidden_states= torch.Size([6, 384, 768])
attention_mask= torch.Size([6, 1, 1, 384])
layer_head_mask= None
encoder_hidden_states= None
encoder_attention_mask= None
past_key_value= None
output_attentions= False
==BertLayer===
===BertAttention===
===BertSelfAttention===
mixed_query_layer= torch.Size([6, 384, 768])
query_layer= torch.Size([6, 12, 384, 64])
query_layer= torch.Size([6, 12, 384, 64])
query_layer= torch.Size([6, 12, 384, 64])
attention_scores.size()= torch.Size([6, 12, 384, 384])
attention_probs.size()= torch.Size([6, 12, 384, 384])
attention_probs.size()= torch.Size([6, 12, 384, 384])
context_layer.size()= torch.Size([6, 12, 384, 64])
context_layer.size()= torch.Size([6, 384, 12, 64])
context_layer.size()= torch.Size([6, 384, 768])
attention_output= torch.Size([6, 384, 768])
attention_output.size()= torch.Size([6, 384, 768])
layer_output= torch.Size([6, 384, 768])
hidden_states= torch.Size([6, 384, 768])
layer_outputs= <class 'tuple'>
len(layer_outputs)= 1
type(layer_outputs)= <class 'tuple'>
i= <class 'torch.Tensor'> len= 6
i size()= torch.Size([6, 384, 768])
==========
=====i- 9 =====
hidden_states= torch.Size([6, 384, 768])
attention_mask= torch.Size([6, 1, 1, 384])
layer_head_mask= None
encoder_hidden_states= None
encoder_attention_mask= None
past_key_value= None
output_attentions= False
==BertLayer===
===BertAttention===
===BertSelfAttention===
mixed_query_layer= torch.Size([6, 384, 768])
query_layer= torch.Size([6, 12, 384, 64])
query_layer= torch.Size([6, 12, 384, 64])
query_layer= torch.Size([6, 12, 384, 64])
attention_scores.size()= torch.Size([6, 12, 384, 384])
attention_probs.size()= torch.Size([6, 12, 384, 384])
attention_probs.size()= torch.Size([6, 12, 384, 384])
context_layer.size()= torch.Size([6, 12, 384, 64])
context_layer.size()= torch.Size([6, 384, 12, 64])
context_layer.size()= torch.Size([6, 384, 768])
attention_output= torch.Size([6, 384, 768])
attention_output.size()= torch.Size([6, 384, 768])
layer_output= torch.Size([6, 384, 768])
hidden_states= torch.Size([6, 384, 768])
layer_outputs= <class 'tuple'>
len(layer_outputs)= 1
type(layer_outputs)= <class 'tuple'>
i= <class 'torch.Tensor'> len= 6
i size()= torch.Size([6, 384, 768])
==========
=====i- 10 =====
hidden_states= torch.Size([6, 384, 768])
attention_mask= torch.Size([6, 1, 1, 384])
layer_head_mask= None
encoder_hidden_states= None
encoder_attention_mask= None
past_key_value= None
output_attentions= False
==BertLayer===
===BertAttention===
===BertSelfAttention===
mixed_query_layer= torch.Size([6, 384, 768])
query_layer= torch.Size([6, 12, 384, 64])
query_layer= torch.Size([6, 12, 384, 64])
query_layer= torch.Size([6, 12, 384, 64])
attention_scores.size()= torch.Size([6, 12, 384, 384])
attention_probs.size()= torch.Size([6, 12, 384, 384])
attention_probs.size()= torch.Size([6, 12, 384, 384])
context_layer.size()= torch.Size([6, 12, 384, 64])
context_layer.size()= torch.Size([6, 384, 12, 64])
context_layer.size()= torch.Size([6, 384, 768])
attention_output= torch.Size([6, 384, 768])
attention_output.size()= torch.Size([6, 384, 768])
layer_output= torch.Size([6, 384, 768])
hidden_states= torch.Size([6, 384, 768])
layer_outputs= <class 'tuple'>
len(layer_outputs)= 1
type(layer_outputs)= <class 'tuple'>
i= <class 'torch.Tensor'> len= 6
i size()= torch.Size([6, 384, 768])
==========
=====i- 11 =====
hidden_states= torch.Size([6, 384, 768])
attention_mask= torch.Size([6, 1, 1, 384])
layer_head_mask= None
encoder_hidden_states= None
encoder_attention_mask= None
past_key_value= None
output_attentions= False
==BertLayer===
===BertAttention===
===BertSelfAttention===
mixed_query_layer= torch.Size([6, 384, 768])
query_layer= torch.Size([6, 12, 384, 64])
query_layer= torch.Size([6, 12, 384, 64])
query_layer= torch.Size([6, 12, 384, 64])
attention_scores.size()= torch.Size([6, 12, 384, 384])
attention_probs.size()= torch.Size([6, 12, 384, 384])
attention_probs.size()= torch.Size([6, 12, 384, 384])
context_layer.size()= torch.Size([6, 12, 384, 64])
context_layer.size()= torch.Size([6, 384, 12, 64])
context_layer.size()= torch.Size([6, 384, 768])
attention_output= torch.Size([6, 384, 768])
attention_output.size()= torch.Size([6, 384, 768])
layer_output= torch.Size([6, 384, 768])
hidden_states= torch.Size([6, 384, 768])
layer_outputs= <class 'tuple'>
len(layer_outputs)= 1
type(layer_outputs)= <class 'tuple'>
i= <class 'torch.Tensor'> len= 6
i size()= torch.Size([6, 384, 768])
==========
===================================
```
### 模型輸出

B x S x N (N為labels數量,此處N=2 開頭與結尾位置)

start_logits, end_logits = logits.split(1, dim=-1)

split後， B x S x 1, B x S x 1

B = 滑動窗格

S = 文本限制長度
```
outputs= QuestionAnsweringModelOutput(loss=None, start_logits=tensor([[ 3.7707,  1.8556, -2.6553,  ..., -5.0599, -5.4542, -1.0339],第一個滑動窗格答案的開頭= torch.Size([384])
        [ 7.0443,  2.0190, -2.6037,  ..., -4.9922, -3.8354, -5.0916],
        [ 3.0287, -1.9691,  2.0251,  ..., -3.3387, -4.7227, -1.9691],
        [ 5.6982, -1.8764,  2.6760,  ..., -4.6843, -4.8519, -4.9672],
        [ 3.0287, -1.9691,  2.0251,  ..., -3.3387, -4.7227, -1.9691],
        [ 5.6982, -1.8764,  2.6760,  ..., -4.6843, -4.8519, -4.9672]]), 第五個滑動窗格的答案的開頭= torch.Size([384])
        end_logits=tensor([[ 3.0410, -1.3288, -1.8149,  ..., -3.6171, -1.2050, -1.6403],第一個滑動窗格答案的結尾= torch.Size([384])
        [ 6.7805, -1.4386, -1.8257,  ..., -3.1951, -2.3751, -3.8283],
        [ 1.8462, -2.4117, -1.8782,  ..., -4.5623, -4.1334, -2.4117],
        [ 5.3498, -2.3074, -0.4370,  ..., -3.8910, -3.9439, -4.2232],
        [ 1.8462, -2.4117, -1.8782,  ..., -4.5623, -4.1334, -2.4117],
        [ 5.3498, -2.3074, -0.4370,  ..., -3.8910, -3.9439, -4.2232]]), hidden_states=None, attentions=None)第五個滑動窗格答案的結尾= torch.Size([384])
```
兩個滑動窗格的開頭與結尾位置
```
======example_index 0 ======
---feature.unique_id= 1000000000 ---#第一個滑動窗格
_get_best_indexes=
feature.unique_id最佳答案開頭位置 [187, 0, 177, 23, 340, 331, 28, 22, 44, 25, 123, 305, 7, 17, 263, 183, 121, 51, 1, 42]
_get_best_indexes=
feature.unique_id最佳答案結尾位置 [119, 303, 0, 338, 40, 211, 30, 101, 120, 224, 176, 175, 304, 339, 186, 181, 253, 21, 84, 14]
version_2_with_negative= False
---feature.unique_id= 1000000001 ---#第二個滑動窗格
_get_best_indexes=
feature.unique_id最佳答案開頭位置 [0, 59, 212, 237, 49, 24, 244, 17, 232, 344, 7, 177, 203, 1, 367, 39, 242, 215, 55, 99]
_get_best_indexes=
feature.unique_id最佳答案結尾位置 [0, 48, 235, 236, 83, 211, 22, 210, 23, 228, 175, 47, 321, 96, 221, 176, 69, 322, 107, 125]
``` 
### 評估方法
匹配出來有20種結果
```
--- 0 ---
output= OrderedDict([('text', '江 苏 路 、 愚 园 路 、 长 宁 路 、 武 夷 路 、 武 定 西 路 、 延 安 西 路'), ('probability', 0.16917861311557703), ('start_logit', 4.777704238891602), ('end_logit', 2.7461767196655273)])
--- 1 ---
output= OrderedDict([('text', '上 海 市 轨 道 交 通 二 号 线 经 过 该 街 道 辖 区 ， 设 有 江 苏 路 站'), ('probability', 0.14705317265714052), ('start_logit', 3.830235242843628), ('end_logit', 3.5534849166870117)])
--- 2 ---
output= OrderedDict([('text', '上 海 市 轨 道 交 通 二 号 线 经 过 该 街 道 辖 区 ， 设 有 江 苏 路 站 。'), ('probability', 0.1132178579092908), ('start_logit', 3.830235242843628), ('end_logit', 3.2920045852661133)])
--- 3 ---
output= OrderedDict([('text', '上 海 市 轨 道 交 通 二 号 线 经 过 该 街 道 辖 区'), ('probability', 0.0617228681228537), ('start_logit', 3.830235242843628), ('end_logit', 2.68534517288208)])
--- 4 ---
output= OrderedDict([('text', '江 苏 路 站'), ('probability', 0.052834535143850996), ('start_logit', 2.8066060543060303), ('end_logit', 3.5534849166870117)])
--- 5 ---
output= OrderedDict([('text', '中 国 上 海 市 长 宁 区 下 辖 的 一 个 街 道 办 事 处'), ('probability', 0.0518543311917725), ('start_logit', 3.578995704650879), ('end_logit', 2.762368679046631)])
--- 6 ---
output= OrderedDict([('text', '中 国 上 海 市 长 宁 区'), ('probability', 0.04564918950664702), ('start_logit', 3.578995704650879), ('end_logit', 2.634916067123413)])
--- 7 ---
output= OrderedDict([('text', '上 海 市 轨 道 交 通 二 号 线'), ('probability', 0.04184587022846674), ('start_logit', 3.830235242843628), ('end_logit', 2.2966837882995605)])
--- 8 ---
output= OrderedDict([('text', '江 苏 路 站 。'), ('probability', 0.040677890755657156), ('start_logit', 2.8066060543060303), ('end_logit', 3.2920045852661133)])
--- 9 ---
output= OrderedDict([('text', '长 宁 区 下 辖 的 一 个 街 道 办 事 处'), ('probability', 0.03710771511837365), ('start_logit', 3.24438214302063), ('end_logit', 2.762368679046631)])
--- 10 ---
output= OrderedDict([('text', '长 宁 区'), ('probability', 0.032667225295658214), ('start_logit', 3.24438214302063), ('end_logit', 2.634916067123413)])
--- 11 ---
output= OrderedDict([('text', '是 中 国 上 海 市 长 宁 区 下 辖 的 一 个 街 道 办 事 处'), ('probability', 0.031003252028768986), ('start_logit', 3.0646493434906006), ('end_logit', 2.762368679046631)])
--- 12 ---
output= OrderedDict([('text', '上 海 市 长 宁 区 下 辖 的 一 个 街 道 办 事 处'), ('probability', 0.027691211133394976), ('start_logit', 2.951672315597534), ('end_logit', 2.762368679046631)])
--- 13 ---
output= OrderedDict([('text', '是 中 国 上 海 市 长 宁 区'), ('probability', 0.027293251974449713), ('start_logit', 3.0646493434906006), ('end_logit', 2.634916067123413)])
--- 14 ---
output= OrderedDict([('text', '江 苏 路 街 道 的 主 要 街 道'), ('probability', 0.02477984186994328), ('start_logit', 3.631213903427124), ('end_logit', 1.9717425107955933)])
--- 15 ---
output= OrderedDict([('text', '上 海 市 长 宁 区'), ('probability', 0.02437754601485415), ('start_logit', 2.951672315597534), ('end_logit', 2.634916067123413)])
--- 16 ---
output= OrderedDict([('text', '江 苏 路 街 道'), ('probability', 0.02040429585256218), ('start_logit', 3.631213903427124), ('end_logit', 1.7774574756622314)])
--- 17 ---
output= OrderedDict([('text', '江 苏 路 街 道 是 中 国 上 海 市 长 宁 区 下 辖 的 一 个 街 道 办 事 处'), ('probability', 0.018722422890430645), ('start_logit', 2.560279130935669), ('end_logit', 2.762368679046631)])
--- 18 ---
output= OrderedDict([('text', '江 苏 路 街 道 是 中 国 上 海 市 长 宁 区'), ('probability', 0.01648200663099981), ('start_logit', 2.560279130935669), ('end_logit', 2.634916067123413)])
--- 19 ---
output= OrderedDict([('text', '上 海 市 的 交 通 干 道 。'), ('probability', 0.015436902559307924), ('start_logit', 2.171422243118286), ('end_logit', 2.9582645893096924)])
```
