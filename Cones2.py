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
cv2.imwrite('images/result.png', output['output_imgs'][0])
print('Image saved to result.png')