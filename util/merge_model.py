import torch
import json, os, shutil
from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM
from typing import Optional

#===================================================================================

def merge_lora_model(lora_checkpoint_dir:str, base_model_dir:Optional[str], merged_model_dir:Optional[str], save_merged_model:bool=False, return_merged_model:bool=True, torch_dtype:torch.dtype = torch.bfloat16, device_map:dict = {"": 0}):
    """
    Merge LoRA adapter and the base model, optionally save the merged model to local directory or return the merged model.

    `AutoPeftModelForCausalLM.from_pretrained` is a method that loads a pre-trained model (LoRA adapters) and its base model.
     The adapter model is loaded from `lora_checkpoint_dir`, which is the directory where the fine-tuned LoRA adapters were saved.
    `AutoPeftModelForCausalLM.from_pretrained` will get the pre-trained base model from the path recorded in `adapter_config.json` in the lora_checkpoint_dir.

    Args:
        lora_checkpoint_dir (str): the directory where the fine-tuned LoRA adapters were saved
        base_model_dir (str, optional): the path to the base model (for the original tokenizer to perserve vocab size for further model  merging)
        merged_model_dir (str, optional): the directory to which the merged model will be saved. Only used if save_merged_model==True
        save_merged_model (bool, optional): If True, will save the merged model to merged_model_dir. Defaults to False.
        return_merged_model (bool, optional): If True, will return the merged model. Defaults to True.
        torch_dtype (torch.dtype, optional): torch dtype used for merging the model. Defaults to torch.bfloat16
        device_map (dict): device map for merging the model. Defaults to {"": 0}

    Returns: the merged model and tokenizer (if return_merged_model==True, or None otherwise)

    """

    new_model = AutoPeftModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=lora_checkpoint_dir,  # this contains info of the base model dir
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch_dtype,
        trust_remote_code=False,
        device_map=device_map,
    )
    merged_model = new_model.merge_and_unload()  # merges the model and unloads it from memory.

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(lora_checkpoint_dir) # use this will cause smaller vocab size
    # tokenizer = AutoTokenizer.from_pretrained(base_model_dir)

    print(f'\nLoaded LoRA adapter and tokenizer from \033[32m{lora_checkpoint_dir}\033[0m and merged with base model  :)\n')

    if save_merged_model:
        # Save the merged model.
        merged_model.save_pretrained(merged_model_dir, trust_remote_code=False, safe_serialization=True)
        # Save the tokenizer.
        tokenizer.save_pretrained(merged_model_dir)

        #------------------------------------------------------------
        # add chat_template to tokenizer_config.json
        # (because .../site-packages/transformers/tokenization_utils_base.py `save_pretrained()` saves chat_template into separate jinja file)
        # ------------------------------------------------------------
        config_file = os.path.join(merged_model_dir, "tokenizer_config.json")
        with open(config_file) as f:
            cfg = json.load(f)
        if "chat_template" not in cfg:
            cfg["chat_template"] = tokenizer.chat_template
        with open(config_file, "w") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
        #------------------------------------------------------------

        # ------------------------------------------------------------
        # Copy license and notice files from base model & adaptor directory
        # ------------------------------------------------------------
        if base_model_dir and os.path.exists(base_model_dir):
            for filename in os.listdir(base_model_dir):
                filename_lower = filename.lower()
                if 'notice' in filename_lower or 'licence' in filename_lower or 'license' in filename_lower:
                    src_path = os.path.join(base_model_dir, filename)
                    new_filename = "BASE_MODEL_" + filename
                    dst_path = os.path.join(merged_model_dir, new_filename)
                    if os.path.isfile(src_path):
                        shutil.copy2(src_path, dst_path)
                        print(f"Copied {filename} to merged model directory")

        if lora_checkpoint_dir and os.path.exists(lora_checkpoint_dir):
            for filename in os.listdir(lora_checkpoint_dir):
                filename_lower = filename.lower()
                if 'notice' in filename_lower or 'licence' in filename_lower or 'license' in filename_lower:
                    src_path = os.path.join(lora_checkpoint_dir, filename)
                    dst_path = os.path.join(merged_model_dir, filename)
                    if os.path.isfile(src_path):
                        shutil.copy2(src_path, dst_path)
                        print(f"Copied {filename} to merged model directory")
        # ------------------------------------------------------------

        print(f"\n\nDONE, model saved to \033[33m{merged_model_dir}\033[0m :)\n")
    if return_merged_model:
        return merged_model, tokenizer
    else:
        return None

# #===============================================
# # Usagge example:
#
# from transformers import AutoTokenizer, pipeline
# from util.prompt import all_prompt_dict
#
# finetuned_model_id = 'V4.2_fewshot_phi4-mini_train200_test200_multitask_cascadedOrder_promptV1_fewshot_R32'
# lora_output_dir = f"checkpoint_dir/experiment/{finetuned_model_id}/"
#
# merged_model_dir = f"model/LORA_SFT/merged_lora_{finetuned_model_id}"
#
# generation_args = {"max_new_tokens": 1024,
#                    "return_full_text": False,
#                    # "temperature": 0.0,
#                    "do_sample": False
#                    }
#
# #---------------------------------------------------
# merged_model, tokenizer = merge_lora_model(lora_output_dir, merged_model_dir, save_merged_model=False, return_merged_model=True)
#
# pipe = pipeline("text-generation", model=merged_model, tokenizer=tokenizer, torch_dtype={"": 0})
#
# user_content = f"{all_prompt_dict['V1_zeroshot']['ASQE']}\n### Input: ```He's very patient but his exams are too hard```"
#
# messages = [{"role": "system", "content": "You are an expert in aspect-based sentiment analysis."},
#                         {"role": "user", "content": user_content}]
#
# response = pipe(messages, **generation_args)[0]['generated_text']
# print(response)