# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from typing import List
import json
from scipy import stats
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))



def infer_stream(engine: 'InferEngine', infer_request: 'InferRequest'):
    request_config = RequestConfig(max_tokens=512, temperature=0, stream=True)
    metric = InferStats()
    gen = engine.infer([infer_request], request_config, metrics=[metric])
    query = infer_request.messages[0]['content']
    # print(f'query: {query}\nresponse: ', end='')

    response_content = ""
    for resp_list in gen:
        if resp_list[0] is None:
            continue
        response_content += resp_list[0].choices[0].delta.content
    #     print(resp_list[0].choices[0].delta.content, end='', flush=True)
    # print()
    # print(f'metric: {metric.compute()}')

    return response_content


if __name__ == '__main__':
    # from swift.llm import load_dataset
    # from swift.llm.argument import TrainArguments
    # import torch
    # from torch.utils.data import DataLoader

    # from qwen2_5_vl import Qwen2_5_VLForConditionalGeneration as Qwen2_5_VL_MLP
    # from utils.template_base import Template
    # from utils.utils import LazyLLMDataset

    # base_model = "/home/wuyuyang/.cache/modelscope/hub/models/Qwen/Qwen2___5-VL-7B-Instruct/"
    # eval_path = "../MLP_datasets/eval_quality.json"
    # args = TrainArguments(model=base_model, dataset=eval_path)
    
    # _, processor = args.get_model_processor()

    # model_path = ""
    # model = Qwen2_5_VL_MLP.from_pretrained(
    #             model_path,
    #             torch_dtype="bfloat16",
    #             device_map="auto"
    # )

    # _, val_dataset = load_dataset(eval_path, split_dataset_ratio=1.0)
    
    # template = args.get_template(processor)
    # my_template = Template(processor, template.template_meta)

    # eval_dataset = LazyLLMDataset(val_dataset, my_template.encode, strict=True, random_state=42)
    
    # data_collator = my_template._data_collator
    # eval_dataloader = DataLoader(
    #     eval_dataset,
    #     batch_size=1,
    #     collate_fn=data_collator,
    # )

    # # Evaluate
    # pred_level_list = []
    # pred_mos_list = []
    # mapping = {"bad": 1, "poor": 2, "fair": 3, "good": 4, "excellent": 5}
    # with torch.no_grad():
    #     for input in tqdm(eval_dataloader):
    #         outputs = model(**input)
    #         pred_mos_list.append(float(outputs["score1"]))

    #         preds = outputs["logits"].argmax(dim=-1).tolist()
    #         output_text = processor.decode(preds[0])
    #         print(output_text)
    #         break
        
    #         found_flag = False
    #         if "quality" in output_text:
    #             for key in mapping:
    #                 if key in output_text:
    #                     pred_level_list.append(float(mapping[key]))
    #                     found_flag = True
    #                     break

    #         if not found_flag:
    #             pred_level_list.append(0)


    # # Compute the metrics
    # labels_qua = []
    # with open("./testset.json", 'r') as f:
    #     datas = json.load(f)
        
    #     for data in datas:
    #         labels_qua.append(float(data['Quality']))

    # print("------------------------------ QA QUALITY ------------------------------")
    # SRCC = stats.spearmanr(pred_level_list, labels_qua)[0] 
    # PLCC = stats.pearsonr(pred_level_list, labels_qua)[0] 
    # KRCC = stats.kendalltau(pred_level_list, labels_qua)[0]
    # print("SRCC: ", SRCC)
    # print("PLCC: ", PLCC)
    # print("KRCC: ", KRCC)

    # print("------------------------------ MLP QUALITY ------------------------------")
    # SRCC = stats.spearmanr(pred_mos_list, labels_qua)[0] 
    # PLCC = stats.pearsonr(pred_mos_list, labels_qua)[0] 
    # KRCC = stats.kendalltau(pred_mos_list, labels_qua)[0]
    # print("SRCC: ", SRCC)
    # print("PLCC: ", PLCC)
    # print("KRCC: ", KRCC)



    from swift.llm import InferEngine, InferRequest, PtEngine, RequestConfig, load_dataset
    from swift.plugin import InferStats
    from qwen2_5_vl import Qwen2_5_VLForConditionalGeneration as Qwen2_5_VL_MLP





    base_model = "/home/wuyuyang/.cache/modelscope/hub/models/Qwen/Qwen2___5-VL-7B-Instruct/"
    model_path = ""
    model = Qwen2_5_VL_MLP.from_pretrained(
                model_path,
                torch_dtype="bfloat16",
                device_map="auto"
    )
    engine = PtEngine(base_model, max_batch_size=64)

    with open("./QA_datasets/eval_quality.json", 'r') as f:
        datas = json.load(f)

    pred_list = []
    for data in tqdm(datas):
        messages = [data["messages"][0]]
        videos = [data["videos"]]
        reply = infer_stream(engine, InferRequest(messages=messages, videos=videos))
        print(messages)
        print(reply)
        break

