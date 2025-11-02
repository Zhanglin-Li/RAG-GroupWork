# 📚 RAG 文献检索系统

reference：[RAGAnything](https://github.com/HKUDS/RAG-Anything) 
## 🔄 RAG流程
简易流程：
<div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%); border-radius: 15px; padding: 25px; margin: 20px 0; border: 2px solid #00d9ff; box-shadow: 0 0 30px rgba(0, 217, 255, 0.3);">

<img src="assets/rag.png" alt="RAG-simple" />

</div>

repo 采用的RAGAnything 框架流程：
<div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%); border-radius: 15px; padding: 25px; margin: 20px 0; border: 2px solid #00d9ff; box-shadow: 0 0 30px rgba(0, 217, 255, 0.3);">

<img src="assets/rag_anything_framework.png" alt="RAG-Anything" />

</div>

### 📋 流程描述：（下面仅考虑纯文本 query 情形，多模态 query 暂不考虑）

完整rag流程详见 `examples/raganything_example.py`的`process_with_rag`函数

0.  参数设置，rag 实例创建(需要config+llm+vlm+embedding)
```python
config = RAGAnythingConfig(...) # 一些功能参数设置
def llm_model_func(...) # 调用 llm 的函数，用于最终获得 answer（context 仅包括文本时）
def vision_model_func(...) # 调用 vlm 的函数，用于描述 doc 图片以及最终获得 answer（context 包括多模态信息时）
embedding_func = EmbeddingFunc(...) # 调用嵌入模型的函数，用于将 query/doc 内容转换为 vector
rag = RAGAnything(config,llm_model_func,vlm_model_func,embedding_func) 
# 整合了整个 rag 流程的实例
```

1. 根据上述参数，解析已有文档并构建知识图谱（KG）及向量数据库（VDB）。对应`rag.process_document_complete`方法。

2. 用户向 KG 和 VDB 发送检索query，之后收到从KG和VDB检索出的context。在`rag.aquery`中实现。

3. 用户将 context 整理为发送给 LLM 的 prompt,发送 prompt 至 LLM/VLM，收到大模型反馈的 answer。在`rag.aquery`中实现。

注：`rag.aquery`实际调用了`rag.aquery_vlm_enhanced`

## ⚡ 流程优化

###  ❓问题：
-  如何确保检索出的 context 对 query 的回答有所帮助？
-  如何以合理的方式向 LLM/VLM 提问以得到更好的答案？

从问题出发，可从以下两个阶段考虑优化 RAG：

###  🔍 检索优化：
####  目标：检索出更"好"的 context

####  分析

得到 context 要经历:
-  使用 doc 构建数据库；
-  用 query 从数据库中检索相关内容。

从而可以考虑在以下方面优化检索：
-  构建数据库
-  query检索

####  数据库构建优化

初始化 rag 实例时，
```python
class RAGAnything(  
lightrag: LightRAG | None = None,  
llm_model_func: ((...) -> Any) | None = None,  
vision_model_func: ((...) -> Any) | None = None,  
embedding_func: ((...) -> Any) | None = None,  
config: RAGAnythingConfig | None = None,  
lightrag_kwargs: Dict[str, Any] = dict  
)
```
`lightrag_kwargs`（调节 `chunk_token_size`等参数）与`config`中包含了可调节的参数供优化。详见`raganything/raganything.py`

####  query 检索优化：

此时需要重写`raganything/query.py`中 `QueryMixin` 的`aquery/aquery_vlm_enhanced`方法。当提供 `vision_model_func`作为 `RAGAnything` 的参数时（此处需要提供，因为文献往往带有图片），`aquery` 方法实际上调用了`aquery_vlm_enhanced`方法，所以需要重写 `enhanced` 方法的逻辑。`enhanced` 方法底层调用的`self.lightrag.aquery`默认做了以下三件事从而会直接返回 llm 答案：检索 `context`，context 包装为 prompt，提交 prompt至llm。但是在检索优化中往往需要获取中间从数据库检索到的 context 用于优化 context 结构或评估检索质量，故需要仿照`QueryMixin`中的 `aquery_vlm_enhanced`重写一个新方法 `aquery_context_optimized` ，在原方法中
```python
# 1. Get original retrieval prompt (without generating final answer)
query_param = QueryParam(mode=mode, only_need_prompt=True, **kwargs)
raw_prompt = await self.lightrag.aquery(query, param=query_param)
```
的后续要提取字符串`raw_prompt`的 `context`部分，处理 `context` 后再与其余部分合并。
raw_prompt格式如下：
```
---Role---
...
---Goal---
...
---Instructions---
...
---Context---
...
---User Query---
...
```


#### 📌 注：
-  re-ranking 功能可以通过在rag 实例初始化时在参数`lightrag_kwargs`中加入 `rerank_model_func`实现。例：
```python
rag = RAGAnything(
...
lightrag_kwargs={
"rerank_model_func": bge_rerank, # 使用本地 rerank
"top_k": 20, # 初步检索 20 个}
)
```


### ✨ 生成优化

####  目标：给定检索到的 context，找到合理利用 context 辅助 llm/vlm 生成"好"答案的途径。

####  分析：

在检索优化完成的基础上，生成优化需要对prompt进行处理，并按特定逻辑发送至 llm。需要在检索优化得到的新方法aquery_context_optimized的基础上再进行修改，得到aquery_prompt_optimized函数。


## 🚀 Quick Start

首先fork此仓库`https://github.com/Zhanglin-Li/RAG-GroupWork`

未安装 uv 则需先安装：
```bash
pip install uv
```
clone fork 后的仓库，并使用 uv 搭建环境：
```bash
# 注意 your_name 要替换为你的github用户名！
git clone https://github.com/your_name/RAG-GroupWork.git
cd RAG-GroupWork
uv sync
```
准备要提供的 document，假设doc路径为/path/to/your/doc，运行脚本：
```bash
uv run ./raganything_example.py /path/to/your/doc
--api-key "your_api_key" \
--base-url "your_base_url" \
-o ./output \
```
其中 api-key 和 base-url 需要从对应 llm/vlm 厂商获取，raganything_example.py 中采用的配置如下，也可替换为其他：
```python
# llm&vlm model config
llm_model_name = "qwen-plus"
vlm_model_name = "qwen-vl-max"
embedding_model_name = "text-embedding-v3"
embedding_dim = 1024
max_token_size = 8192
```


## 💡 提示

- 原仓库的 example 中rag.process_document_complete方法没有指定minerU parser 的 source='modelscope'，在不翻墙情况下会有网络问题， 显式指定即可解决。
- 一次解析完成会将KG 和 VDB 储存在 ./rag_storage 目录下，后续处理同一批文档时会直接使用，不重复解析。
- 文档处理（解析和数据库构建）非常慢，cpu 处理仅一篇 pdf 并构建数据库可能需要数分钟，（或许可用 gpu 但未尝试），可先用一篇简单pdf试验，再在若干文献上试验。
