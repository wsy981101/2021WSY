![Python 3.8](https://img.shields.io/badge/python-3.8-green.svg)


### Overview
![图1](https://github.com/wsy981101/TAM-ULAG/assets/95170179/39b42479-8b6b-4056-a089-2d4292790453)
Abstract: Most multimodal emotion recognition methods aim to find an effective fusion mechanism to construct the features from heterogeneous modalities, so as to learn the feature representation with semantic consistency. However, these methods usually ignore the emotionally semantic differences between modalities. To solve this problem, one multi-task learning framework is proposed. By training one multimodal task and three unimodal tasks jointly, the emotionally semantic consistency information among multimodal features and the emotionally semantic difference information contained in each modality are respectively learned. Firstly, in order to learn the emotionally semantic consistency information, one Temporal Attention Mechanism (TAM) based on a multilayer recurrent neural network is proposed. The contribution degree of emotional features is described by assigning different weights to time series feature vectors. Then, for multimodal fusion, the fine-grained feature fusion per semantic dimension is carried out in the semantic space. Secondly, one self-supervised Unimodal Label Automatic Generation (ULAG) strategy based on the inter-modal feature vector similarity is proposed in order to effectively learn the difference information of emotional semantics in each modality. A large number of experimental results on three datasets, CMU-MOSI, CMU-MOSEI, CH-SIMS, confirm that the proposed TAM-ULAG model has strong competitiveness, and has improved the classification indices ( ,  ) and regression index (MAE, Corr) compared with the current benchmark models. For binary classification, the recognition rate is 87.2% and 85.8% on the CMU-MOSEI and CMU-MOSEI datasets, and 81.47% on the CH-SIMS dataset. The results show that simultaneously learning the emotionally semantic consistency information and the emotionally semantic difference information for each modality is helpful in improving the performance of self-supervised multimodal emotion recognition method.

### Data availability

1. Datasets and pre-trained berts

Download dataset features and pre-trained berts from the following links.
- [Google Cloud Drive](https://drive.google.com/drive/folders/1E5kojBirtd5VbfHsFp6FYWkQunk73Nsv?usp=sharing)

For all features, you can use `SHA-1 Hash Value` to check the consistency.
> `MOSI/unaligned_50.pkl`: `5da0b8440fc5a7c3a457859af27458beb993e088`  
> `MOSI/aligned_50.pkl`: `5c62b896619a334a7104c8bef05d82b05272c71c`  
> `MOSEI/unaligned_50.pkl`: `db3e2cff4d706a88ee156981c2100975513d4610`  
> `MOSEI/aligned_50.pkl`: `ef49589349bc1c2bc252ccc0d4657a755c92a056`  
> `SIMS/unaligned_39.pkl`: `a00c73e92f66896403c09dbad63e242d5af756f8`  

Due to the size limitations, the MOSEI features and SIMS raw videos are available in `Baidu Cloud Drive` only. All dataset features are organized as:

```python
{
    "train": {
        "raw_text": [],
        "audio": [],
        "vision": [],
        "id": [], # [video_id$_$clip_id, ..., ...]
        "text": [],
        "text_bert": [],
        "audio_lengths": [],
        "vision_lengths": [],
        "annotations": [],
        "classification_labels": [], # Negative(< 0), Neutral(0), Positive(> 0)
        "regression_labels": []
    },
    "valid": {***}, # same as the "train" 
    "test": {***}, # same as the "train"
}
```

For MOSI and MOSEI, the pre-extracted text features are from BERT, different from the original glove features in the [CMU-Multimodal-SDK](http://immortal.multicomp.cs.cmu.edu/raw_datasets/processed_data/).

For SIMS, if you want to extract features from raw videos, you need to install [Openface Toolkits](https://github.com/TadasBaltrusaitis/OpenFace/wiki) first, and then refer our codes in the `data/DataPre.py`.

```
python data/DataPre.py --data_dir [path_to_Dataset] --language ** --openface2Path  [path_to_FeatureExtraction]
```

For bert models, you also can download [Bert-Base, Chinese](https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip) from [Google-Bert](https://github.com/google-research/bert). And then, convert tensorflow into pytorch using [transformers-cli](https://huggingface.co/transformers/converting_tensorflow_models.html)  

2. Clone this repo and install requirements.
```
git clone https://github.com/wsy981101/TAM-ULAG.git
cd TAM-ULAG
conda create --name tam_ulag python=3.8
source activateTAM-ULAG
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f 
pip install pytorch_transformers
```

3. Make some changes
Modify the `config/config_tune.py` and `config/config_regression.py` to update dataset pathes.

4. Run codes
```
python run.py --modelName ta_gml
```

### Contact
If you have any questions, please feel free to reach me out at qsun@xaut.edu.cn and 2210320080@stu.xaut.edu.cn.


### Citation
If this work is useful to you, please cite:

   Sun, Q., Wang, S.Y.: Self-supervised Multimodal Emotion Recognition Based on Temporal Attention Mechanism and Unimodal Label Automatic Generation Strateg[J]. Journal of Electronics & Information Technology. doi: 10.11999/JEIT231107.

Qiang Sun<sup>1,2</sup>,Shuyu Wang<sup>1</sup>
1.	Department of Communication Engineering, School of Automation and Information Engineering, Xi’an University of Technology, Xi’an 710048, China
2.	Xi’an Key Laboratory of Wireless Optical Communication and Network Research, Xi’an 710048, China

