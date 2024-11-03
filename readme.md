实现个性化地多概念定制化文本到图像生成。通过对Stable-Diffusion模型中的text-encoder部分进行定制化微调，即可得到特定概念的定制化残差。通过组合多个不同概念的定制化残差并利用预先指定的布局设计引导生成过程，即可在同一场景中定制你想要的多种专属概念！

模型描述
本模型基础的Diffusion Model采用Stable-Diffusion-v1-5预训练模型，经过训练后得到的定制化残差存储仅需要5-6KB。

期望模型使用方式以及适用范围
如何使用
基于 ModelScope 框架 pip install -r requirements.txt 通过调用预定义的 Pipeline 可实现快速调用。

快速模型推理
通过以下代码即可快速实现Cones2的推理过程。

import cv2
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

task = Tasks.text_to_image_synthesis
model_id = 'damo/Cones2'
pipe = pipeline(task=task, model=model_id, model_revision='v1.0')
output = pipe({'text': 'a mug and a dog on the beach',
               "residual_dict": {
                   "dog": "../../modelscope/pipelines/multi_modal/cone2_pipeline/residuals/dog.pt",
                   "mug": "../../modelscope/pipelines/multi_modal/cone2_pipeline/residuals/mug.pt",
               },
               "subject_list": [["mug", 2], ["dog", 5]],
               "color_context": {"255,192,0": ["mug", 2.5], "255,0,0": ["dog", 2.5]},
               "layout": "../../data/test/cones2/mask_example.png"})
cv2.imwrite('result.png', output['output_imgs'][0])
print('Image saved to result.png')
相关论文以及引用信息
如果该模型对您有所帮助，请引用下面的相关的论文：

@article{liu2023cones,
  title={Cones 2: Customizable Image Synthesis with Multiple Subjects},
  author={Liu, Zhiheng and Zhang, Yifei and Shen, Yujun and Zheng, Kecheng and Zhu, Kai and Feng, Ruili and Liu, Yu and Zhao, Deli and Zhou, Jingren and Cao, Yang},
  journal={arXiv preprint arXiv:2305.19327},
  year={2023}
}