# Your Attention is Unique: Detecting 360-Degree Video Saliency in Head-Mounted Display for Head Movement Prediction
This repository contains the source code for the PanoSalnet model in the ACMMM18 paper:

Anh Nguyen, Zhisheng Yan, and Klara Nahrstedt 2018. Your Attention is Unique: Detecting 360-Degree Video Saliency in Head-Mounted Display for Head Movement Prediction. In ACM Multimedia Conference for 2018 (ACMMM2018)

To cite our paper, use this Bitex code: 
```
@inproceedings{panosalnet2018,
  author = {Nguyen, Anh and Yan, Zhisheng and Nahrstedt, Klara},
  title = {{Your Attention is Unique: Detecting 360-Degree Video Saliency in Head-Mounted Display for Head Movement Prediction}},
  booktitle = {ACM Multimedia Conference for 2018 (ACMMM2018)},
  year = {2018}
}
```
# Abstract
Head movement prediction is the key enabler for the emerging 360-degree videos since it can enhance both streaming and rendering efficiency. To achieve accurate head movement prediction, it becomes imperative to understand user's visual attention on 360-degree videos under head-mounted display (HMD). Despite the rich history of saliency detection research, we observe that traditional models are designed for regular images/videos fixed at a single viewport and would introduce problems such as central bias and multi-object confusion when applied to the multi-viewport 360-degree videos switched by user interaction. To fill in this gap, this paper shifts the traditional single-viewport saliency models that have been extensively studied for decades to a fresh panoramic saliency detection specifically tailored for 360-degree videos, and thus maximally enhances the head movement prediction performance. The proposed head movement prediction framework is empowered by a newly created dataset for 360-degree video saliency, a panoramic saliency detection model and an integration of saliency and head tracking history for the ultimate head movement prediction. Experimental results demonstrate the measurable gain of both the proposed panoramic saliency detection and head movement prediction over traditional models for regular images/videos.

## Saliency groundtruth generation.

## PanoSalNet
![panosalnet-fig](https://raw.githubusercontent.com/phananh1010/PanoSalNet/5b7cbdf334b6699d0a11e4a2aa2dee541f430f41/360net.pdf)



# Contact
If you have any general doubt about our work, please use the [public issues section](repohttps://github.com/phananh1010/PanoSalNet/issues) on this github. Alternatively, drop us an e-mail at <mailto:anguyen139@student.gsu.edu> or <mailto:zyan@gsu.edu>.
