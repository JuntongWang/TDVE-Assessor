# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from typing import List
import json
from scipy import stats
from tqdm import tqdm

def infer_batch(engine: 'InferEngine', infer_requests: List['InferRequest']):
    pred_list = []

    request_config = RequestConfig(max_tokens=512, temperature=0)
    metric = InferStats()
    resp_list = engine.infer(infer_requests, request_config, metrics=[metric])

    query0 = infer_requests[0].messages[0]['content']
    print(f'query0: {query0}')
    print(f'response0: {resp_list[0].choices[0].message.content}')
    print(f'metric: {metric.compute()}')
    # metric.reset()  # reuse

    for i in range(len(resp_list)):
        pred_list.append(resp_list[i].choices[0].message.content)

    return pred_list


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

#----------------------------------------------------------------------

if __name__ == '__main__':
    from swift.llm import InferEngine, InferRequest, PtEngine, RequestConfig, load_dataset
    from swift.plugin import InferStats
    # model = 'Qwen/Qwen2.5-1.5B-Instruct'
    model = '/home/wooyiyang/ms-swift-main/output/v3-20250225-145522/checkpoint-3200-merged/'
    infer_backend = 'pt'

    if infer_backend == 'pt':
        engine = PtEngine(model, max_batch_size=64)
    elif infer_backend == 'vllm':
        from swift.llm import VllmEngine
        engine = VllmEngine(model, max_model_len=32768)
    elif infer_backend == 'lmdeploy':
        from swift.llm import LmdeployEngine
        engine = LmdeployEngine(model)

    # Here, `load_dataset` is used for convenience; `infer_batch` does not require creating a dataset.
    # dataset = load_dataset(['./test_1.json'], seed=42)[0]
    # print(f'dataset: {dataset}')
    # infer_requests = [InferRequest(**data) for data in dataset]
    # pred_list = infer_batch(engine, infer_requests)

    with open('../../datasets/video_mos2_test.json', 'r') as f:
        datas = json.load(f)
    
    pred_list = []
    labels_list = []
    eval_info = {}
    for data in tqdm(datas):
        label = data["messages"][1]["content"]
        labels_list.append(label)
        messages = [data["messages"][0]]
        videos = [data["videos"]]
        # messages = [{'role': 'user', 'content': '<video>Please rate the quality of the human face in this video, considering factors such as resolution, clarity, smoothness, and overall visual quality.'}]
        reply = infer_stream(engine, InferRequest(messages=messages, videos=videos))
        eval_info[data["videos"]] = reply
        if reply == label:
            pred_list.append(reply)
    

    print("------------------------------ ACCURACY ------------------------------")
    print(len(pred_list) / len(labels_list))

    with open("qa_eval_info.json", 'w') as f:
        json.dump(eval_info, f)