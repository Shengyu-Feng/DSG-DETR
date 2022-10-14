# Dynamic Scene Graph Detection Transformer (DSG-DETR)
Pytorch implementation of [Exploiting Long-Term Dependencies for Generating Dynamic Scene Graphs], WACV 2023. Our code uses the [Spatial-Temporal Transformer for Dynamic Scene Graph Generation, ICCV 2021] repository (https://github.com/yrcong/STTran) as a baseline. 


**About the code**
We run the code on a single  NVIDIA Tesla V100S GPU for both training and testing. We borrowed some code from [Yuren's repository](https://github.com/yrcong/STTran)  ,  [Yang's repository](https://github.com/jwyang/faster-rcnn.pytorch) and [Zellers' repository](https://github.com/rowanz/neural-motifs).

## Usage
We use python=3.7.0, pytorch=1.9 and torchvision=0.10.0 in our code. First, clone this repository:
<!-- ```git clone https://github.com/Shengyu-Feng/STSG.git ``` --> 
We borrow some compiled code for bbox operations.
```
cd lib/draw_rectangles
python setup.py build_ext --inplace
cd ..
cd fpn/box_intersections_cpu
python setup.py build_ext --inplace
```
For the object detector part, please follow the compilation from https://github.com/jwyang/faster-rcnn.pytorch
We provide a pretrained FasterRCNN model for Action Genome. Please download [here](https://drive.google.com/file/d/1-u930Pk0JYz3ivS6V_HNTM1D5AxmN5Bs/view?usp=sharing) and put it in 
```
fasterRCNN/models/faster_rcnn_ag.pth
```

## Dataset
We use the dataset [Action Genome](https://www.actiongenome.org/#download) to train/evaluate our method. Please process the downloaded dataset with the [Toolkit](https://github.com/JingweiJ/ActionGenome). The directories of the dataset should look like:
```
|-- action_genome
    |-- annotations   #gt annotations
    |-- frames        #sampled frames
    |-- videos        #original videos
```
 In the experiments for SGCLS/SGDET, we only keep bounding boxes with short edges larger than 16 pixels. Please download the file [object_bbox_and_relationship_filtersmall.pkl](https://drive.google.com/file/d/19BkAwjCw5ByyGyZjFo174Oc3Ud56fkaT/view?usp=sharing) and put it in the ```dataloader```

## Train
You can train the **DSG-DETR** with train.py. We trained the model on a TESLA V100S:
+ For PredCLS: 
```
python train.py -mode predcls -datasize large -data_path $DATAPATH 
```
+ For SGCLS: 
```
python train.py -mode sgcls -datasize large -data_path $DATAPATH 
```
+ For SGDET: 
```
python train.py -mode sgdet -datasize large -data_path $DATAPATH 
```

## Evaluation
You can evaluate the **DSG-DETR** with test.py.
+ For PredCLS ([trained Model](https://drive.google.com/file/d/18oFR8hfH3W84AYjR1yktsjQKeIlKbilo/view?usp=sharing)): 
```
python test.py -m predcls -datasize large -data_path $DATAPATH -model_path $MODELPATH
```
+ For SGCLS ([trained Model](https://drive.google.com/file/d/1E3fTGyh7Uhcsy7nBfrrY0t3jIi88uclF/view?usp=sharing)): : 
```
python test.py -m sgcls -datasize large -data_path $DATAPATH -model_path $MODELPATH
```
+ For SGDET ([trained Model](https://drive.google.com/file/d/19qW2x61eXBhQ2x3liJSRmKOF6zKqtYjV/view?usp=sharing)): : 
```
python test.py -m sgdet -datasize large -data_path $DATAPATH -model_path $MODELPATH
```


## Citation

If you find our work helpful, please cite our paper:

```
@misc{https://doi.org/10.48550/arxiv.2112.09828,
  doi = {10.48550/ARXIV.2112.09828},
  
  url = {https://arxiv.org/abs/2112.09828},
  
  author = {Feng, Shengyu and Tripathi, Subarna and Mostafa, Hesham and Nassar, Marcel and Majumdar, Somdeb},
  
  keywords = {Computer Vision and Pattern Recognition (cs.CV), FOS: Computer and information sciences, FOS: Computer and information sciences},
  
  title = {Exploiting Long-Term Dependencies for Generating Dynamic Scene Graphs},
  
  publisher = {arXiv},
  
  year = {2021},
  
  copyright = {arXiv.org perpetual, non-exclusive license}
}

```
