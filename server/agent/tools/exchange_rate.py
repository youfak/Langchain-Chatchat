## 这个模型的提示词非常复杂，我们推荐使用GPT4模型进行运行
from __future__ import annotations

## 单独运行的时候需要添加
# import sys
# import os
#
# sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import re
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

## 使用https://app.exchangerate-api.com/ 汇率转换
KEY = "98682ce64dbda8432a0297b3"

_PROMPT_TEMPLATE = """
用户会提出一个关于汇率的问题，你的目标是拆分出用户问题中的货币并转成对应的货币代码按照我提供的工具回答，答案请使用中文。
例如 用户提出的问题是: 人民币转美元的汇率是多少？
则 提取的货币是: None 人民币(CNY) 美元(USD)
如果用户提出的问题是: 人民币,美元的汇率？
则 提取的货币是: None 人民币(CNY) 美元(USD)
如果用户提出的问题是: 人民币兑美元的汇率？
则 提取的货币是: None 人民币(CNY) 美元(USD)
如果用户提出的问题是: 1元人民币兑美元是多少钱？
则 提取的货币是: 1 人民币(CNY) 美元(USD)
如果用户提出的问题是: 5元人民币可以换多少美元?
则 提取的货币是: 5 人民币(CNY) 美元(USD)
如果用户提出的问题是: 1块人民币兑美元?
则 提取的货币是: 1 人民币(CNY) 美元(USD)

请注意以下内容:
1. 如果你没有找到具体数值的内容,则一定要使用 None 替代，否则程序无法运行
2. 如果没有提取到货币信息 则直接返回缺少信息

问题: ${{用户的问题}}

你的回答格式应该按照下面的内容，请注意，格式内的```text 等标记都必须输出，这是我用来提取答案的标记。
```text

${{拆分的货币，中间用空格隔开}}
```
... exchangerate(数值 货币 货币)...
```output

${{提取后的答案}}
```
答案: ${{答案}}



这是一个例子：
问题: 人民币转美元的汇率是多少？
这是一个例子：
问题: 1块人民币兑美元?

```text
1 人民币(CNY) 美元(USD)
```
...exchangerate(1 人民币(CNY) 美元(USD))...

```output
一块人民币可以换0.732美元

Answer: 1块人民币转美元等于多少。

现在，这是我的问题：

问题: {question}
"""
PROMPT = PromptTemplate(
    input_variables=["question"],
    template=_PROMPT_TEMPLATE,
)


def get_rate_info(currency, key):
    # Making our request
    base_url = f'https://v6.exchangerate-api.com/v6/{key}/latest/{currency}'
    response = requests.get(base_url)
    data = response.json()
    return data


def rate(query):
    key = KEY
    if key == "":
        return "请先在代码中填入Exchangerate API Key"
    try:
        currency = ""
        parts = query.split()
        pattern = r"\((\w+)\)"
        match_obj = re.search(pattern, parts[1])
        if match_obj:
            currency = match_obj.group(1)
            print(currency)
        data = get_rate_info(currency, key)
        if data["result"] == "error":
            return f"调用接口查询汇率失败，请检查你的输入是否正确 解析结果:{query}\n"
        data = data["conversion_rates"]
        rate_type = re.search(pattern, parts[2]).group(1)
        if parts[0] != "None":
            count = float(parts[0]) * float(data[rate_type])
            return f"{count}\n"
        return f"{parts[1]}兑{parts[2]}汇率: {data[rate_type]}\n"

    except KeyError:
        try:

            return "重要提醒：你好\n"
        except KeyError:
            return "汇率信息分析错误"


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
            output = rate(expression)
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
            return {self.output_key: f"输入的格式不对: {llm_output},应该输入 (货币兑货币或货币,货币)的组合"}
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


def exchangerate(query: str):
    model = model_container.MODEL
    llm_rate = LLMRateChain.from_llm(model, verbose=True, prompt=PROMPT)
    ans = llm_rate.run(query)
    return ans


class ExchangeRateSchema(BaseModel):
    location: str = Field(description="应该是一个汇率转换或换算，用空格隔开，例如：人民币 美元，如果没有数值的信息，可以只输入货币名称如:人民币 美元，如果有则输入数值和货币名称，例如：1 人民币 美元")


if __name__ == '__main__':
    # result = exchangerate("一块人民币转美元等于多少？")
    key = KEY
    get_rate_info(key)
