from langchain.tools import Tool
from server.agent.tools import *

## 请注意，如果你是为了使用AgentLM，在这里，你应该使用英文版本。

tools = [
    Tool.from_function(
        func=calculate,
        name="calculate",
        description="Useful for when you need to answer questions about simple calculations",
        args_schema=CalculatorInput,
    ),
    Tool.from_function(
        func=arxiv,
        name="arxiv",
        description="A wrapper around Arxiv.org for searching and retrieving scientific articles in various fields.",
        args_schema=ArxivInput,
    ),
    Tool.from_function(
        func=weathercheck,
        name="weather_check",
        description="",
        args_schema=WhetherSchema,
    ),
    Tool.from_function(
        func=shell,
        name="shell",
        description="Use Shell to execute Linux commands",
        args_schema=ShellInput,
    ),
    Tool.from_function(
        func=search_knowledgebase_complex,
        name="search_knowledgebase_complex",
        description="Use Use this tool to search local knowledgebase and get information",
        args_schema=KnowledgeSearchInput,
    ),
    Tool.from_function(
        func=search_internet,
        name="search_internet",
        description="Use this tool to use bing search engine to search the internet",
        args_schema=SearchInternetInput,
    ),
    Tool.from_function(
        func=wolfram,
        name="Wolfram",
        description="Useful for when you need to calculate difficult formulas",
        args_schema=WolframInput,
    ),
    Tool.from_function(
        func=search_youtube,
        name="search_youtube",
        description="use this tools to search youtube videos",
        args_schema=YoutubeInput,
    ),
    Tool.from_function(
        func=exchangerate,
        name="汇率查询工具",
        description="如果用户输入是汇率相关问题，这个工具可以帮你解答问题",
        args_schema=ExchangeRateSchema,
    ),
    Tool.from_function(
        func=search_yellow_pages,
        name="search_yellow_pages",
        description="""这个是一个收录了澳门本地店铺信息的大型数据库，收录了澳门店铺的店铺名称、行业分类、行业细类、地址、電話、傳真、電子郵件、網址、Facebook、微信、廣告、簡介、分區、營業時間、巴士路線、经纬度坐标信息。
首先，数据库中的每个店铺都有一个唯一的店铺名称，这是店铺的正式名称，可以准确地识别和查找店铺。这个名称可能包括店铺的品牌、类型、地点等信息，能够快速理解店铺的基本情况。
其次，每个店铺都被归入一个特定的行业分类，这有助于了解店铺的主要业务类型。这个分类是根据澳门的行业分类来确定的，可以快速筛选出店铺所在的行业类型。
数据库还记录了每个店铺的经营范围，这是对店铺主要经营项目和服务的详细描述，可以据此来了解店铺的主要业务内容。店铺的经营范围信息可能记录于数据库里的行业分类、行业细类、店铺名称、廣告、簡介这几个字段当中。
每个店铺的具体地址也被记录下来，包括所属澳门哪个分区、街道名称、门牌号码等，能够精准找到店铺的具体位置。
数据库还提供了每个店铺的联系方式，包括电话、传真、电子邮件和微信号，这些信息可以直接与相关店铺进行联系和交流。
数据库还提供了店铺官方网址、Facebook网址，点击这些网址可以更进一步地了解店铺的特色信息。
每个店铺的简介和广告词也被记录下来，这些信息可以帮助了解店铺的基本情况和主要宣传点。这些信息可能包括店铺的经营范围、发展历史、特色、优势等内容。
数据库还记录了每个店铺所在的澳门地理分区，这是对店铺所在地理区域或商圈的描述，这个分区信息是根据店铺的实际位置来确定的。
每个店铺的营业时间也被记录下来，包括每天的开门和关门时间，还有每周或每个月的哪一天休息。这些信息可以帮助规划访问店铺的时间。这个时间是精确到分钟的。
数据库还提供了到达每个店铺的巴士路线信息，这些信息包括店铺附件的巴士站名称和店铺到这些巴士站的距离，还列举了经过这些巴士站的巴士线路，可以帮助规划到达此店铺的公共交通路线。
最后，数据库还收录了每个店铺的经纬度坐标，在地图上可以更加直观地了解店铺的地址位置，可以根据用户当前的经纬度坐标和查询信息来推荐最近的店铺、规划到店路线等。
总的来说，这个数据库是一个全面且详细的澳门店铺信息数据库，它提供了丰富的信息，可以满足用户对澳门店铺信息的各种查询需求。
""",
        args_schema=YellowPagesInput,)
]

tool_names = [tool.name for tool in tools]
