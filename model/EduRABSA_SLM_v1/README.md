[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg


# Licence and Copyright

[![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa] &ensp; [![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]


* The "**EduRABSA_SLM**" LoRA adaptors and any merged models derived from them are licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].


* Copyright (c) 2025 Authors of [Data-Efficient Adaptation and a Novel Evaluation Method for Aspect-based Sentiment Analysis](https://arxiv.org/abs/2511.03034).


* The two pre-trained base models ("the original models") [Phi4-mini-instruct](https://huggingface.co/microsoft/Phi-4-mini-instruct) and [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) were used for inference and training LoRA adaptors. No modifications were made to the original models. The original models’ licences and copyright notices (where provided) are included in the corresponding subdirectories of this repository.

# The EduRABSA_SLM Model Family

The "**EduRABSA_SLM**" model family consists of fine-tuned multi-task small LLMs (SLMs) designed for resource-efficient opinion mining on education-domain reviews of courses, teaching staff, and universities (e.g. student course or teaching evaluations, and open-ended survey responses).

To cite the LoRA adaptors or merged models in this family:

```
@misc{hua2025dataefficientadaptationnovelevaluation,
      title={Data-Efficient Adaptation and a Novel Evaluation Method for Aspect-based Sentiment Analysis}, 
      author={Yan Cathy Hua and Paul Denny and Jörg Wicker and Katerina Taškova},
      year={2025},
      eprint={2511.03034},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2511.03034}, 
}
```


## Model Info

The **EduRABSA_SLM** multi-task models can perform opinion mining across the following fine-grained Aspect-based Sentiment Analysis (ABSA) tasks, extracting outputs for and across each review entry as illustrated in the image below:

* Opinion Extraction (OE)
* Aspect-Opinion Pair-Extraction (AOPE)
* Aspect-opinion Categorisation (AOC; ASC with opinion term)
* Aspect-(opinion)-Sentiment Triplet Extraction (ASTE)
* Aspect-(opinion-category)-Sentiment Quadruplet Extraction (ASQE)

&emsp;&emsp;  <img src="https://github.com/yhua219/ftsobp_and_edurabsa_slm/blob/main/.figs/fig_table_1.png?raw=true" alt="ABSA examples" width="850px;">



* The **EduRABSA_SLM** LoRA adaptors were fine-tuned for two pre-trained models: [Phi4-mini-instruct](https://huggingface.co/microsoft/Phi-4-mini-instruct) (3.8B parameters) and [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) (1.5B parameters), using the [EduRABSA dataset](https://arxiv.org/abs/2508.17008). 
 

* Full details regarding the development and performance of the **EduRABSA_SLM** models are available in our paper 
[Data-Efficient Adaptation and a Novel Evaluation Method for Aspect-based Sentiment Analysis](https://arxiv.org/abs/2511.03034).


* The fine-tuning process used [this example script](https://huggingface.co/microsoft/Phi-4-mini-instruct/blob/main/sample_finetune.py). The training parameters and training logs are available at [./_training_log](_training_log), where each training log folder contains two files:
  - Full training parameters and training log:  `training_summary.txt`.
  - Tensorboard log file: `"events.out.tfevents....."`,

    
## 🤗 Huggingface Download Link

* The EduRABSA_SLM models and LoRA adaptors are available on Huggingface 🤗 in [the EduRABSA SLM collection](https://huggingface.co/collections/yhua219/edurabsa-slm)

## How to Use

* To merge the LoRA adaptors with the base model, please see [./scripts/1_merge_lora_adaptor_with_base_model.ipynb](../../scripts/1_merge_lora_adaptor_with_base_model.ipynb).

* To use the LoRA adaptor with or without saving the merged model, please see [./scripts/2a_lora_model_inference_and_eval_with_FTS_OBP.ipynb](../../scripts/2a_lora_model_inference_and_eval_with_FTS_OBP.ipynb).

* To use an entire model (in this case Huggingface models), please see [./scripts/2b_merged_model_inference.ipynb](../../scripts/2b_merged_model_inference.ipynb) or [./scripts/2c_merged_model_inference_and_eval_with_FTS_OBP.ipynb](../../scripts/2c_merged_model_inference_and_eval_with_FTS_OBP.ipynb).
