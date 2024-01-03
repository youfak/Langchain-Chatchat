## 这个模型的提示词非常复杂，我们推荐使用GPT4模型进行运行
from __future__ import annotations

import random
## 单独运行的时候需要添加
# import sys
# import os
#
# sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import re
import time
import warnings
from typing import Dict

from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.pydantic_v1 import Extra, root_validator
from langchain.schema import BasePromptTemplate
from langchain.schema.language_model import BaseLanguageModel
import requests
from typing import List, Any, Optional
from langchain.prompts import PromptTemplate
from server.agent import model_container
from pydantic import BaseModel, Field

host = 'http://192.168.50.215:5000'

_PROMPT_TEMPLATE = """
用户会提出一个关于澳门的相关问题，你的目标是拆分出用户问题中的行业分类、行业细类、店铺名称、地址、电话、传真、邮箱、官网、Facebook、微信、广告、簡介、分區、营业时间、bus线路、坐标转成数据库对应的属性按照我提供的工具回答，你不用特意增加属性字段，你需要对结果整理出正确的答案。
Category(行业分类)只能在下面选择一个:
飲食業,購物,醫療健康及美容,住宿,旅遊及娛樂,教育,金融及保險,運輸及通訊,專業及商業服務,社會服務及公共事務,傳媒與通訊

下面是举例:
例如 用户提出的问题是: Grand Lisboa Hotel在那里
则 提取的内容是: StoreName|Grand Lisboa Hotel;Address|None
如果用户提出的问题是: 咖啡室
则 提取的内容是: IndustryCategories|咖啡室
如果用户提出的问题是: 澳门有哪些地方可以吃饭？
则 提取的内容是: Category|飲食業
如果用户提出的问题是: 澳门有哪些中式餐厅？
则 提取的内容是: IndustryCategories|中式餐厅
如果用户提出的问题是: 澳门假日酒店地址在那里？
则 提取的内容是: StoreName|假日酒店;Address|None
如果用户提出的问题是: 澳门假日酒店电话是多少？
则 提取的内容是: StoreName|假日酒店;Telephone|电话
如果用户提出的问题是: 澳门有那些药店
则 提取的内容是: Category|医疗

请注意以下内容:
1. 一定要把拆分出来内容转为繁体，否则我无法为你提供答案
2. 请取的内容之间使用分号(;)隔开
3. 小括号内的内容是必选的
4. 行业分类=Category、行业细类=IndustryCategories、店铺名称=StoreName、地址=Address、电话=Telephone、传真=Fax、邮箱=Mail、官网=Url、Facebook=Facebook、微信=WeChat、广告=Advertise、簡介=Introduction、分區=Partition、营业时间=StartBusinessHours、bus线路=BusRoute
5. 如果没有提取到内容 则直接返回缺少信息

问题: ${{用户的问题}}

你的回答格式应该按照下面的内容，请注意，格式内的```text 等标记都必须输出，这是我用来提取答案的标记。
```text

${{拆分的内容，中间用|隔开}}
```
... search_yellow_pages(Category|餐厅;IndustryCategories|西餐厅)...
```output

${{提取后的答案}}
```
答案: ${{答案}} ---该答案是来自于黄页数据库



这是一个例子：
问题: 澳门有那些西餐厅？

```text
IndustryCategories|西餐厅
```
...search_yellow_pages(IndustryCategories|西餐厅)...

```output

Answer: 米其林一星餐厅 ---该答案是来自于黄页数据库

现在，这是我的问题：

问题: {question}
"""
PROMPT = PromptTemplate(
    input_variables=["question"],
    template=_PROMPT_TEMPLATE,
)


def yellow_pages_search(query):
    # Making our request
    base_url = f'{host}/yellow/trends'
    response = requests.post(base_url, headers={
        'Content-Type': 'application/json',
        'Accept': '*/*',
    }, json={"data": query, "type": "info"})  # 发送post请求，第一个参数是URL，第二个参数是请求数据
    data = response.text
    if data == "":
        time.sleep(random.randint(1, 2))
        return ""
    return data


def get_yellow_pages_fuzzy_search(query):
    # Making our request
    base_url = f'{host}/yellow/fuzzy_search'
    response = requests.post(base_url, headers={
        'Content-Type': 'application/json',
        'Accept': '*/*',
    }, json={"data": query, "type": "info"})  # 发送post请求，第一个参数是URL，第二个参数是请求数据
    data = response.text
    if data == "":
        time.sleep(random.randint(1, 2))
        return "输入信息有误，请尝试其他属性"
    return data


def yellow_pages(query):
    try:
        return get_yellow_pages_fuzzy_search(query)
    except KeyError:
        return "信息分析错误"


class LLMRateChain(Chain):
    llm_chain: LLMChain
    llm: Optional[BaseLanguageModel] = None
    """[Deprecated] LLM wrapper to use."""
    prompt: BasePromptTemplate = PROMPT
    """[Deprecated] Prompt to use to translate to python if necessary."""
    input_key: str = "question"  #: :meta private:
    output_key: str = "answer"  #: :meta private:

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @root_validator(pre=True)
    def raise_deprecation(cls, values: Dict) -> Dict:
        if "llm" in values:
            warnings.warn(
                "Directly instantiating an LLMWeatherChain with an llm is deprecated. "
                "Please instantiate with llm_chain argument or using the from_llm "
                "class method."
            )
            if "llm_chain" not in values and values["llm"] is not None:
                prompt = values.get("prompt", PROMPT)
                values["llm_chain"] = LLMChain(llm=values["llm"], prompt=prompt)
        return values

    @property
    def input_keys(self) -> List[str]:
        """Expect input key.

        :meta private:
        """
        return [self.input_key]

    @property
    def output_keys(self) -> List[str]:
        """Expect output key.

        :meta private:
        """
        return [self.output_key]

    def _evaluate_expression(self, expression: str) -> str:
        try:
            output = yellow_pages(expression)
        except Exception as e:
            output = "输入的信息有误，请再次尝试"
        return output

    def _process_llm_result(
            self, llm_output: str, run_manager: CallbackManagerForChainRun
    ) -> Dict[str, str]:

        run_manager.on_text(llm_output, color="green", verbose=self.verbose)

        llm_output = llm_output.strip()
        text_match = re.search(r"^```text(.*?)```", llm_output, re.DOTALL)
        if text_match:
            expression = text_match.group(1)
            output = self._evaluate_expression(expression)
            run_manager.on_text("\nAnswer: ", verbose=self.verbose)
            run_manager.on_text(output, color="yellow", verbose=self.verbose)
            answer = "Answer: " + output
        elif llm_output.startswith("Answer:"):
            answer = llm_output
        elif "Answer:" in llm_output:
            answer = "Answer: " + llm_output.split("Answer:")[-1]
        else:
            return {self.output_key: f"信息有误无法解析: {llm_output}"}
        return {self.output_key: answer}

    async def _aprocess_llm_result(
            self,
            llm_output: str,
            run_manager: AsyncCallbackManagerForChainRun,
    ) -> Dict[str, str]:
        await run_manager.on_text(llm_output, color="green", verbose=self.verbose)
        llm_output = llm_output.strip()
        text_match = re.search(r"^```text(.*?)```", llm_output, re.DOTALL)

        if text_match:
            expression = text_match.group(1)
            output = self._evaluate_expression(expression)
            await run_manager.on_text("\nAnswer: ", verbose=self.verbose)
            await run_manager.on_text(output, color="yellow", verbose=self.verbose)
            answer = "Answer: " + output
        elif llm_output.startswith("Answer:"):
            answer = llm_output
        elif "Answer:" in llm_output:
            answer = "Answer: " + llm_output.split("Answer:")[-1]
        else:
            raise ValueError(f"unknown format from LLM: {llm_output}")
        return {self.output_key: answer}

    def _call(
            self,
            inputs: Dict[str, str],
            run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        _run_manager.on_text(inputs[self.input_key])
        llm_output = self.llm_chain.predict(
            question=inputs[self.input_key],
            stop=["```output"],
            callbacks=_run_manager.get_child(),
        )
        return self._process_llm_result(llm_output, _run_manager)

    async def _acall(
            self,
            inputs: Dict[str, str],
            run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        _run_manager = run_manager or AsyncCallbackManagerForChainRun.get_noop_manager()
        await _run_manager.on_text(inputs[self.input_key])
        llm_output = await self.llm_chain.apredict(
            question=inputs[self.input_key],
            stop=["```output"],
            callbacks=_run_manager.get_child(),
        )
        return await self._aprocess_llm_result(llm_output, _run_manager)

    @property
    def _chain_type(self) -> str:
        return "llm_rate_chain"

    @classmethod
    def from_llm(
            cls,
            llm: BaseLanguageModel,
            prompt: BasePromptTemplate = PROMPT,
            **kwargs: Any,
    ) -> LLMRateChain:
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        return cls(llm_chain=llm_chain, **kwargs)


def search_yellow_pages(query: str):
    model = model_container.MODEL
    llm_rate = LLMRateChain.from_llm(model, verbose=True, prompt=PROMPT)
    ans = llm_rate.run(query)
    return ans


class YellowPagesInput(BaseModel):
    location: str = Field(description="搜索黄页数据库")


if __name__ == '__main__':
    result = search_yellow_pages("澳门有那些西餐厅？")
    print("答案:", result)
