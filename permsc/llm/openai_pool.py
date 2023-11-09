__all__ = ['ChatCompletionPool', 'OpenAIConfig']

import logging
from dataclasses import dataclass
import multiprocessing as mp
import time
from typing import List, Type, Any

import openai
from openai import ChatCompletion, InvalidRequestError
from openai.api_resources.abstract.engine_api_resource import EngineAPIResource
from tqdm import tqdm


@dataclass
class OpenAIConfig:
    deployment_name: str = ''
    model_name: str = 'gpt-3.5-turbo'
    api_base: str = ''
    api_key: str = ''
    api_version: str = ''
    api_type: str = 'azure'  # one of 'azure', 'litellm', or 'openai'


class EngineAPIResourcePool:
    openai_resource_class: Type[EngineAPIResource] = None
    batch_key: str = None

    def __init__(self, configs: List[OpenAIConfig]):
        self.model_name = configs[0].model_name
        self.configs = configs
        self.model_queue = mp.Queue()

        for config in configs:
            self.model_queue.put(config)

        self.work_queue = mp.Queue()
        self.result_queue = mp.Queue()
        self.processes = [mp.Process(target=self._entrypoint) for _ in configs]

        for process in self.processes:
            process.start()

    def stop(self):
        for process in self.processes:
            process.kill()

    def _entrypoint(self):
        while work := self.work_queue.get():
            idx, args, kwargs = work
            delay = 2

            while True:
                try:
                    config = self.model_queue.get()

                    try:
                        if config.api_type == 'litellm':
                            openai.api_key = 'litellm'
                            openai.api_base = config.api_base
                            result = openai.ChatCompletion.create(*args, model='litellm', **kwargs)
                        else:
                            api_kwargs = dict(
                                api_key=config.api_key,
                                api_base=config.api_base,
                                api_version=config.api_version,
                                api_type=config.api_type,
                                engine=config.deployment_name,
                                model=config.model_name,
                            )

                            if config.api_type == 'openai':
                                del api_kwargs['engine']  # OpenAI.com doesn't use engine

                            result = self.openai_resource_class.create(*args, **api_kwargs, **kwargs)
                    finally:
                        self.model_queue.put(config)

                    delay = 2
                    break
                except InvalidRequestError:
                    result = None
                    break
                except:
                    import traceback
                    traceback.print_exc()
                    time.sleep(delay)
                    delay = min(delay * 2, 30)

                    if delay == 30:
                        logging.error('Giving up this request.')
                        result = None
                        break

            self.result_queue.put((idx, result))

    def create_batch(self, *args, callback=None, **kwargs) -> List[Any]:
        for idx, kwarg in enumerate(kwargs[self.batch_key]):
            new_kwargs = kwargs.copy()
            new_kwargs[self.batch_key] = kwarg
            self.work_queue.put((idx, args, new_kwargs))

        results = []

        for _ in tqdm(kwargs[self.batch_key]):
            result = self.result_queue.get()

            if callback:
                callback(*result)

            results.append(result)

        return [result for _, result in sorted(results, key=lambda x: x[0])]


class ChatCompletionPool(EngineAPIResourcePool):
    openai_resource_class = ChatCompletion
    batch_key = 'messages'
