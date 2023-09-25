__all__ = ['ChatCompletionPool', 'OpenAIConfig']

from dataclasses import dataclass
import multiprocessing as mp
import time
from typing import List, Type, Any

from openai import ChatCompletion
from openai.api_resources.abstract.engine_api_resource import EngineAPIResource
from tqdm import tqdm


@dataclass
class OpenAIConfig:
    deployment_name: str = ''
    model_name: str = 'gpt-3.5-turbo-16k'
    api_base: str = ''
    api_key: str = ''
    api_version: str = '2023-07-01-preview'
    api_type: str = 'azure'


class EngineAPIResourcePool:
    openai_resource_class: Type[EngineAPIResource] = None
    batch_key: str = None

    def __init__(self, configs: List[OpenAIConfig]):
        self.configs = configs
        self.model_queue = mp.Queue()

        for config in configs:
            self.model_queue.put(config)

        self.work_queue = mp.Queue()
        self.result_queue = mp.Queue()
        self.processes = [mp.Process(target=self._entrypoint) for _ in configs]

        for process in self.processes:
            process.start()

    def _entrypoint(self):
        while work := self.work_queue.get():
            config = self.model_queue.get()
            idx, args, kwargs = work

            try:
                while True:
                    try:
                        result = self.openai_resource_class.create(
                            *args,
                            api_key=config.api_key,
                            api_base=config.api_base,
                            api_version=config.api_version,
                            api_type=config.api_type,
                            engine=config.deployment_name,
                            model=config.model_name,
                            **kwargs
                        )
                        break
                    except:
                        import traceback
                        traceback.print_exc()
                        time.sleep(5)

                self.result_queue.put((idx, result))
            finally:
                self.model_queue.put(config)

    def create_batch(self, *args, **kwargs) -> List[Any]:
        for idx, kwarg in enumerate(kwargs[self.batch_key]):
            new_kwargs = kwargs.copy()
            new_kwargs[self.batch_key] = kwarg
            self.work_queue.put((idx, args, new_kwargs))

        results = []

        for _ in tqdm(kwargs[self.batch_key]):
            results.append(self.result_queue.get())

        return [result for _, result in sorted(results, key=lambda x: x[0])]


class ChatCompletionPool(EngineAPIResourcePool):
    openai_resource_class = ChatCompletion
    batch_key = 'messages'
